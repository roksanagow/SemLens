import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from scripts.guardian_corpora import (GuardianClient,CollectionError,article_usage,
    read_bins,sample_period,search_parameters,export_corpora,local_key)
from semlens.data_loading import load_from_csv

PERIOD={'name':'2020-2021','start':'2020-01-01','end':'2021-12-31'}


def article(i, body=None, **changes):
    result={'id':f'news/2020/test-{i}','type':'article','webPublicationDate':'2020-06-10T12:00:00Z',
            'webTitle':f'Article {i}','webUrl':f'https://www.theguardian.com/news/2020/test-{i}',
            'sectionName':'Society','fields':{'body':body or f'<p>A support bubble helped household {i}.</p>',
                                           'publication':'The Guardian','lastModified':'2025-01-01T00:00:00Z'}}
    result.update(changes)
    return result


def test_exact_targets_paragraphs_inline_markup_and_dates():
    body='<p>Dr. Jones discussed a <b>bub</b>ble. “Two bubbles remain!”</p><p>Another bubble formed.</p><script>bubble</script><figure>bubble caption</figure>'
    usage,record=article_usage(article(1,body),PERIOD,42)
    assert usage['sentence'][usage['start']:usage['end']] in ('bubble','bubbles')
    assert '<b>' not in record['body'] and 'caption' not in record['body']
    assert record['body_occurrences']==3
    assert usage['date'].startswith('2020-') and usage['last_modified'].startswith('2025-')
    assert article_usage(article(2,'<p>The runner bubbled with excitement.</p>'),PERIOD,42) is None
    assert article_usage(article(3,webPublicationDate='2022-01-01T00:00:00Z'),PERIOD,42) is None
    assert article_usage(article(4,type='liveblog'),PERIOD,42) is None
    assert article_usage(article(5,fields={}),PERIOD,42) is None


def test_seeded_sampling_covers_whole_result_set_and_exports_exact_usages(tmp_path):
    class FakeClient:
        def __init__(self): self.pages=[]
        def search(self,params):
            page=params['page']
            self.pages.append(page)
            return {'total':120,'results':[article(i) for i in range((page-1)*50,min(page*50,120))]}
    quiet=lambda *a,**k:None
    client=FakeClient()
    rows,records,rejected=sample_period(client,PERIOD,120,12,42,set(),quiet)
    assert len(rows)==len(records)==12 and rejected==0
    assert len({r['article_id'] for r in rows})==12
    assert len(client.pages)==len(set(client.pages)) and max(client.pages)>1
    again,_,_=sample_period(FakeClient(),PERIOD,120,12,42,set(),quiet)
    assert rows==again
    export_corpora(tmp_path,[PERIOD],rows,records,[],{'seed':42})
    loaded=load_from_csv(tmp_path/'bubble_usages.csv','bubble',forms=['bubbles'])
    assert len(loaded)==12 and loaded.corpus_labels==['2020-2021']
    assert loaded.char_spans==[(r['start'],r['end']) for r in rows]
    assert loaded.sentences==[r['sentence'] for r in rows]
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert 'bubble_usages.csv' in manifest['files']
    assert (tmp_path/'mini_corpora/2020-2021.txt').exists()
    assert len((tmp_path/'articles/2020-2021.jsonl').read_text().splitlines())==12


def test_sampling_skips_duplicate_missing_and_stem_only_hits():
    results=[article(1),article(1),article(2,'<p>They are bubbling.</p>'),article(3),article(4,fields={})]
    client=SimpleNamespace(search=lambda p:{'total':len(results),'results':results})
    rows,records,rejected=sample_period(client,PERIOD,len(results),10,42,set(),lambda *a,**k:None)
    assert len(rows)==2 and rejected==3
    assert {r['article_id'] for r in rows}=={article(1)['id'],article(3)['id']}
    changed=SimpleNamespace(search=lambda p:{'total':20,'results':results})
    with pytest.raises(CollectionError,match='changed during collection'):
        sample_period(changed,PERIOD,5,2,42,set(),lambda *a,**k:None)


def test_api_cache_resumes_without_key_and_never_persists_secret(tmp_path):
    calls=[]
    payload={'response':{'status':'ok','total':1,'results':[article(1)]}}
    def get(url,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200,json=lambda:payload)
    client=GuardianClient(tmp_path,'secret-for-test',session=SimpleNamespace(get=get))
    params=search_parameters(PERIOD)
    assert client.search(params)['total']==1
    assert calls[0]['params']['api-key']=='secret-for-test'
    assert 'api-key' not in params
    assert all('secret-for-test' not in p.read_text() for p in tmp_path.iterdir())
    offline=GuardianClient(tmp_path,None,session=SimpleNamespace(get=lambda *a,**k:pytest.fail('Network called for cached data')))
    assert offline.search(params)['total']==1
    with pytest.raises(CollectionError,match='developer key'):
        offline.search(search_parameters(PERIOD,2))


@pytest.mark.parametrize('status,message',[(401,'rejected'),(429,'quota'),(500,'HTTP 500')])
def test_api_failures_do_not_echo_response_or_key(tmp_path,status,message):
    client=GuardianClient(tmp_path,'secret-for-test',session=SimpleNamespace(get=lambda *a,**k:SimpleNamespace(status_code=status)))
    with pytest.raises(CollectionError,match=message) as exc:
        client.search(search_parameters(PERIOD))
    assert 'secret-for-test' not in str(exc.value)
    assert not list(tmp_path.glob('*.json'))


def test_budget_and_network_errors_are_resumable_and_redacted(tmp_path):
    client=GuardianClient(tmp_path,'secret-for-test',max_requests=0)
    with pytest.raises(CollectionError,match='budget'):
        client.search(search_parameters(PERIOD))
    def fail(*a,**k): raise requests.ConnectionError('https://example.com/?api-key=secret-for-test')
    client=GuardianClient(tmp_path,'secret-for-test',session=SimpleNamespace(get=fail))
    with pytest.raises(CollectionError) as exc:
        client.search(search_parameters(PERIOD))
    assert 'secret-for-test' not in str(exc.value)


def test_bins_validation_and_local_key(tmp_path,monkeypatch):
    import scripts.guardian_corpora as collector
    monkeypatch.setattr(collector,'ROOT',tmp_path)
    monkeypatch.setenv('GUARDIAN_API_KEY','environment-key')
    assert local_key()=='environment-key'
    monkeypatch.delenv('GUARDIAN_API_KEY')
    (tmp_path/'.streamlit').mkdir()
    (tmp_path/'.streamlit/secrets.toml').write_text('guardian_api_key = "local-key"\n')
    assert local_key()=='local-key'
    path=tmp_path/'bins.json'
    path.write_text(json.dumps([PERIOD]))
    assert read_bins(path)==[PERIOD]
    path.write_text(json.dumps([PERIOD,{**PERIOD,'name':'overlap'}]))
    with pytest.raises(CollectionError,match='overlap'):
        read_bins(path)
    path.write_text(json.dumps([{**PERIOD,'name':'../unsafe'}]))
    with pytest.raises(CollectionError,match='names'):
        read_bins(path)


def test_json_key_file_and_downloaded_response_are_distinguished(tmp_path,monkeypatch):
    import scripts.guardian_corpora as collector
    monkeypatch.setattr(collector,'ROOT',tmp_path)
    monkeypatch.setattr(collector.Path,'home',lambda:tmp_path)
    monkeypatch.delenv('GUARDIAN_API_KEY',raising=False)
    (tmp_path/'.streamlit').mkdir()
    path=tmp_path/'.streamlit/guardian-key.json'
    path.write_text(json.dumps({'api_key':'local-json-key'}))
    assert local_key()=='local-json-key'
    path.write_text(json.dumps({'response':{'status':'ok','results':[article(1)]}}))
    with pytest.raises(CollectionError,match='API response, not a key'):
        local_key()
    path.write_text('{"api_key": "do-not-echo-this-key"')
    with pytest.raises(CollectionError) as exc:
        local_key()
    assert 'do-not-echo-this-key' not in str(exc.value)
