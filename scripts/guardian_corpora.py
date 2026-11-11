"""Build small, dated bubble corpora from the Guardian Content API.

Run with the project's Python environment; see docs/guardian-collection.md.
The API key is read locally and is never included in cache keys or manifests.
"""
from __future__ import annotations
import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from semlens.data_loading import find_occurrences

ENDPOINT='https://content.guardianapis.com/search'
QUERY='"bubble" OR "bubbles"'
PAGE_SIZE=50
FIELDS='body,headline,byline,publication,lastModified'
USAGE_FIELDS=['sentence','corpus','start','end','sentence_id','matched_form','article_id',
              'date','title','section','publication','byline','url','last_modified']


class CollectionError(Exception):
    pass


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def local_key():
    key=os.environ.get('GUARDIAN_API_KEY','').strip()
    if key:
        return key
    for path in [ROOT/'.streamlit/secrets.toml',Path.home()/'.streamlit/secrets.toml']:
        if path.is_file():
            try:
                try:
                    import tomllib
                    values=tomllib.loads(path.read_text(encoding='utf-8'))
                except ImportError:  # Python 3.10 with Streamlit's toml dependency.
                    import toml
                    values=toml.loads(path.read_text(encoding='utf-8'))
                key=values.get('guardian_api_key','')
                if isinstance(key,str) and key.strip():
                    return key.strip()
            except Exception:
                raise CollectionError('Could not read the local secrets file. Check its TOML syntax.') from None
    json_path=ROOT/'.streamlit/guardian-key.json'
    if json_path.is_file():
        try:
            values=json.loads(json_path.read_text(encoding='utf-8'))
        except (ValueError,OSError):
            raise CollectionError('Could not read .streamlit/guardian-key.json. Use a JSON object with an api_key field.') from None
        if isinstance(values,str) and values.strip():
            return values.strip()
        if isinstance(values,dict):
            for name in ('api_key','api-key','guardian_api_key'):
                key=values.get(name)
                if isinstance(key,str) and key.strip():
                    return key.strip()
            if isinstance(values.get('response'),dict):
                raise CollectionError('guardian-key.json contains an API response, not a key. Replace it with {"api_key": "YOUR_KEY"}, using the key from your registration email or the api-key parameter of your API URL.')
        raise CollectionError('No API key found in guardian-key.json. Use {"api_key": "YOUR_KEY"}.')
    return None


class GuardianClient:
    def __init__(self, cache, key=None, *, max_requests=400, session=None):
        self.cache=Path(cache)
        self.key=key
        self.max_requests=max_requests
        self.requests=0
        self.session=session or requests.Session()
        self.last_request=None

    def search(self, params):
        fingerprint=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
        path=self.cache/f'{fingerprint}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))['response']
        if not self.key:
            raise CollectionError('A Guardian developer key is needed. Set api_key in .streamlit/guardian-key.json, guardian_api_key in .streamlit/secrets.toml, or GUARDIAN_API_KEY in the environment. Do not put it in this script.')
        if self.requests>=self.max_requests:
            raise CollectionError('Request budget reached. Cached responses are preserved; resume later with the same output directory.')
        if self.last_request is not None:
            time.sleep(max(0,1.05-(time.monotonic()-self.last_request)))
        self.last_request=time.monotonic()
        self.requests+=1
        try:
            response=self.session.get(ENDPOINT,params={**params,'api-key':self.key},timeout=45)
        except requests.RequestException:
            # Requests exceptions can contain the key-bearing URL.
            raise CollectionError('Guardian request failed. Cached responses are preserved; retry with the same output directory.') from None
        if response.status_code in (401,403):
            raise CollectionError(f'Guardian rejected API access (HTTP {response.status_code}). Check the local developer key and its access level.')
        if response.status_code==429:
            raise CollectionError('Guardian rate/quota limit reached (HTTP 429). Stop and resume later; cached requests will not be repeated.')
        if response.status_code!=200:
            raise CollectionError(f'Guardian request failed (HTTP {response.status_code}). No response text or key has been logged.')
        try:
            payload=response.json()['response']
            if payload['status']!='ok' or not isinstance(payload['results'],list):
                raise ValueError()
        except (ValueError,KeyError,TypeError):
            raise CollectionError('Guardian returned an unexpected response; no corpus was exported.') from None
        write_json(path,{'retrieved_at':datetime.now(timezone.utc).isoformat(),'parameters':params,'response':payload})
        return payload


def read_bins(path):
    bins=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(bins,list) or not 1<=len(bins)<=10:
        raise CollectionError('Use a JSON list containing 1–10 time bins.')
    previous=None
    names=set()
    for period in bins:
        name=period['name']
        start,end=date.fromisoformat(period['start']),date.fromisoformat(period['end'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) or name in names:
            raise CollectionError('Time-bin names must be unique and contain only letters, numbers, hyphens and underscores.')
        if start>end or (previous is not None and start<=previous):
            raise CollectionError('Time bins must be chronological and must not overlap.')
        previous=end
        names.add(name)
    return bins


def search_parameters(period, page=1, *, bodies=False):
    params={'q':QUERY,'query-fields':'body','tag':'type/article','lang':'en',
            'from-date':period['start'],'to-date':period['end'],'use-date':'published',
            'order-by':'oldest','order-date':'published','page-size':PAGE_SIZE,'page':page}
    if bodies:
        params['show-fields']=FIELDS
    return params


class BodyText(HTMLParser):
    """Keep paragraph boundaries and inline text, omitting embedded non-prose."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
        self.ignored=[]

    def handle_starttag(self, tag, attrs):
        if tag in {'script','style','figure','aside'}:
            self.ignored.append(tag)
        if not self.ignored and tag in {'p','li','h2','h3','blockquote','br'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if self.ignored:
            if tag==self.ignored[-1]:
                self.ignored.pop()
        elif tag in {'p','li','h2','h3','blockquote'}:
            self.parts.append('\n')

    def handle_data(self, text):
        if not self.ignored:
            self.parts.append(text)

    def paragraphs(self):
        return [clean for line in ''.join(self.parts).splitlines() if (clean:=re.sub(r'\s+',' ',line).strip())]


def sentences(paragraphs):
    """Simple, inspectable sentence splitting; preserve closing quote marks."""
    abbreviations={'mr.','mrs.','ms.','dr.','prof.','st.','jr.','sr.','e.g.','i.e.','u.s.','u.k.'}
    for paragraph in paragraphs:
        start=0
        for match in re.finditer(r'''[.!?]+["'”’)]*(?:\s+|$)''',paragraph):
            prefix=paragraph[start:match.end()].strip()
            last=prefix.split()[-1] if prefix else ''
            if last.lower() in abbreviations or re.fullmatch(r'[A-Z]\.',last):
                continue
            if prefix:
                yield prefix
            start=match.end()
        if paragraph[start:].strip():
            yield paragraph[start:].strip()


def article_usage(article, period, seed):
    if article.get('type')!='article':
        return None
    published=article.get('webPublicationDate','')
    day=published[:10]
    if not (period['start']<=day<=period['end']):
        return None
    fields=article.get('fields') or {}
    parser=BodyText()
    parser.feed(fields.get('body') or '')
    paragraphs=parser.paragraphs()
    if not paragraphs:
        return None
    choices=[]
    article_id=article['id']
    for i,sentence in enumerate(sentences(paragraphs)):
        sid=hashlib.sha256(f'{article_id}\0{i}\0{sentence}'.encode()).hexdigest()
        for start,end in find_occurrences(sentence,'bubble',['bubbles']):
            choices.append({'sentence':sentence,'start':start,'end':end,'sentence_id':sid,
                            'matched_form':sentence[start:end]})
    if not choices:
        return None
    chosen=random.Random(f'{seed}:{article_id}').choice(choices)
    metadata={'article_id':article_id,'date':published,'title':article.get('webTitle',''),
              'section':article.get('sectionName',''),'publication':fields.get('publication',''),
              'byline':fields.get('byline',''),'url':article.get('webUrl',''),
              'last_modified':fields.get('lastModified','')}
    return ({**chosen,'corpus':period['name'],**metadata},
            {**metadata,'corpus':period['name'],'body':'\n\n'.join(paragraphs),'body_occurrences':len(choices)})


def sample_period(client, period, total, limit, seed, seen_ids, progress=print):
    """Visit a seeded random permutation of article ranks, never the first N hits."""
    ranks=list(range(total))
    random.Random(f'{seed}:{period["name"]}').shuffle(ranks)
    pages={}
    usages,articles=[],[]
    rejected=0
    for rank in ranks:
        page,offset=divmod(rank,PAGE_SIZE)
        page+=1
        if page not in pages:
            progress(f'{period["name"]}: fetching sample page {page}; {len(usages)}/{limit} usages retained',flush=True)
            result=client.search(search_parameters(period,page,bodies=True))
            if int(result['total'])!=total:
                raise CollectionError(f'The search results for {period["name"]} changed during collection. Use a new output directory for a fresh snapshot.')
            pages[page]=result['results']
        if offset>=len(pages[page]):
            raise CollectionError('Guardian pagination changed during collection; use a new output directory.')
        article=pages[page][offset]
        if article['id'] in seen_ids:
            rejected+=1
            continue
        item=article_usage(article,period,seed)
        if item is None:
            rejected+=1
            continue
        usage,record=item
        seen_ids.add(article['id'])
        usages.append(usage)
        articles.append(record)
        if len(usages)>=limit:
            break
    return usages,articles,rejected


def csv_text(rows, fields):
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def export_corpora(output, bins, usages, articles, summary, config):
    (output/'mini_corpora').mkdir(parents=True,exist_ok=True)
    (output/'articles').mkdir(parents=True,exist_ok=True)
    (output/'bubble_usages.csv').write_text(csv_text(usages,USAGE_FIELDS),encoding='utf-8')
    for period in bins:
        name=period['name']
        rows=[row for row in usages if row['corpus']==name]
        # Usage-only text for convenience; CSV retains the exact selected target
        # when a sentence contains several instances of bubble/bubbles.
        (output/'mini_corpora'/f'{name}.txt').write_text('\n'.join(row['sentence'] for row in rows)+'\n',encoding='utf-8')
        records=[a for a in articles if a['corpus']==name]
        (output/'articles'/f'{name}.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in records),encoding='utf-8')
    write_json(output/'manifest.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
        'configuration':config,'summary':summary,'sampling':'Uniform random article ranks within each bin; one random target occurrence per retained article.',
        'limitations':['Article-balanced target samples, not representative samples of all newspaper language or all target occurrences.',
                      'Publication dates are webPublicationDate, not lastModified; republication can affect dating.',
                      'Sentence boundaries use a heuristic and should be reviewed before a presentation.',
                      'Publication/section mixtures and unequal time-bin durations can affect comparisons.'],
        'files':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [output/'bubble_usages.csv',*sorted((output/'mini_corpora').glob('*.txt')),*sorted((output/'articles').glob('*.jsonl'))]}})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bins',type=Path,default=Path(__file__).with_name('guardian_bubble_bins.json'))
    parser.add_argument('--output',type=Path,default=ROOT/'.semlens/datasets/guardian_bubble')
    parser.add_argument('--per-bin',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-requests',type=int,default=400)
    parser.add_argument('--counts-only',action='store_true')
    args=parser.parse_args(argv)
    if args.per_bin<1 or not 1<=args.max_requests<=500:
        parser.error('--per-bin must be positive; --max-requests must be between 1 and 500.')
    try:
        bins=read_bins(args.bins)
        config={'version':1,'query':QUERY,'forms':['bubble','bubbles'],'bins':bins,
                'per_bin':args.per_bin,'seed':args.seed,'date_field':'webPublicationDate','page_size':PAGE_SIZE}
        run_file=args.output/'run.json'
        if run_file.exists() and json.loads(run_file.read_text())!=config:
            raise CollectionError('This output directory belongs to a different configuration. Choose a new --output directory.')
        write_json(run_file,config)
        client=GuardianClient(args.output/'cache',local_key(),max_requests=args.max_requests)
        counts=[]
        for period in bins:
            response=client.search(search_parameters(period))
            count={'corpus':period['name'],'start':period['start'],'end':period['end'],'api_matches':int(response['total'])}
            counts.append(count)
            write_json(args.output/'counts.json',counts)
            print(f'{period["name"]}: {count["api_matches"]:,} API matches',flush=True)
        if args.counts_only:
            print(f'Counts saved in {args.output}/counts.json. No article sample exported.')
            return 0
        usages,articles,summary=[],[],[]
        seen=set()
        for period,count in zip(bins,counts):
            rows,records,rejected=sample_period(client,period,count['api_matches'],args.per_bin,args.seed,seen)
            usages.extend(rows)
            articles.extend(records)
            summary.append({**count,'sampled_articles':len(records),'selected_occurrences':len(rows),'rejected_candidates':rejected})
        if not usages:
            raise CollectionError('No usable target occurrences were retrieved. No mini-corpora exported.')
        export_corpora(args.output,bins,usages,articles,summary,config)
        print(f'Exported {len(usages)} usages from {len(articles)} articles to {args.output}/bubble_usages.csv',flush=True)
        print(f'{client.requests} new API requests; cached responses retained for reproducibility and resuming.')
        return 0
    except (CollectionError,OSError,ValueError,KeyError) as exc:
        print(f'Collection stopped: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
