"""Deterministic pure knowledge search and dependency-aware reads."""
from __future__ import annotations
import base64, hashlib, json, re, unicodedata
from pathlib import Path

class RetrievalError(ValueError): pass

def _tokens(text):
    t=unicodedata.normalize('NFKC',text).lower(); words=re.findall(r"[a-z0-9_]+",t)
    # CJK single-character matches are too common to be useful retrieval evidence; use adjacent
    # bigrams plus explicit English/API tokens, as specified by the public index contract.
    runs = re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+", t)
    return words + [run[i:i+2] for run in runs for i in range(len(run)-1)]

class KnowledgeStore:
    def __init__(self, root):
        from .publish import validate_package
        validate_package(root, require_index=True)
        self.root=Path(root); self.cards={p.stem:json.loads(p.read_text()) for p in (self.root/'cards').glob('*.json')}; self.refs={p.stem:json.loads(p.read_text()) for p in (self.root/'reference').glob('*.json')}; self.packs={p.stem:json.loads(p.read_text()) for p in (self.root/'packs').glob('*.json')}
        self.index=json.loads((self.root/'index'/'search.json').read_text()) if (self.root/'index'/'search.json').is_file() else {"items":[]}
        self.bindings=json.loads((self.root/'index'/'mechanism_bindings.json').read_text(encoding='utf-8'))['bindings']
        from .publish import package_fingerprint
        self.package_digest=package_fingerprint(self.root)
    def topics(self):
        return [{'id': key, 'title': pack['title'], 'summary': pack['summary'],
                 'card_count': len(pack['card_ids'])}
                for key, pack in sorted(self.packs.items())]
    def search(self,text,limit=5):
        q=set(_tokens(text)); scored=[]
        for item in self.index.get('items',[]):
            score=5*len(q & set(_tokens(item.get('title','')+' '+' '.join(item.get('aliases',[])))))+3*len(q&set(_tokens(item.get('summary',''))))+2*len(q&set(_tokens(' '.join(item.get('source_features',[])))))+len(q&set(_tokens(item.get('text',''))))
            if score: scored.append((-score,item['id'],item))
        return [x[2] for x in sorted(scored)[:max(0,min(10,limit))]]
    @staticmethod
    def _cursor(payload):
        raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
        body=base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')
        check=hashlib.sha256(raw).hexdigest()[:16]
        return body+'.'+check
    @staticmethod
    def _decode_cursor(value):
        if not isinstance(value,str) or len(value)>2048 or value.count('.')!=1:
            raise RetrievalError('invalid knowledge cursor')
        body,check=value.split('.',1)
        try:
            raw=base64.urlsafe_b64decode(body+'='*((4-len(body)%4)%4))
            payload=json.loads(raw.decode('utf-8'))
        except Exception as exc:
            raise RetrievalError('invalid knowledge cursor') from exc
        if hashlib.sha256(raw).hexdigest()[:16]!=check or type(payload) is not dict:
            raise RetrievalError('invalid knowledge cursor')
        return payload
    def search_page(self, query='', *, observations=(), limit=8, topic=None, cursor=None, browse=False):
        if type(limit) is not int or not 1<=limit<=8:
            raise RetrievalError('knowledge search page size must be between 1 and 8')
        if type(browse) is not bool:
            raise RetrievalError('invalid knowledge browse request')
        topics=self.topics()
        if browse:
            if query or observations or topic is not None or cursor is not None:
                raise RetrievalError('topic browsing cannot be combined with search or a cursor')
            return {'items':[],'next_cursor':None,'total_matches':0,'topics':topics}
        if topic is not None and topic not in self.packs:
            raise RetrievalError('unknown knowledge topic')
        if not isinstance(query,str) or not isinstance(observations,(list,tuple)) or any(not isinstance(x,str) for x in observations):
            raise RetrievalError('invalid knowledge search query')
        combined=' '.join([query,*observations]).strip()
        query_identity={'query':query,'observations':list(observations),'topic':topic,
                        'limit':limit,'sort':'score-desc-id-asc-v1'}
        query_digest=hashlib.sha256(json.dumps(query_identity,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        offset=0
        if cursor is not None:
            payload=self._decode_cursor(cursor)
            if payload.get('kind')!='search' or payload.get('package')!=self.package_digest or payload.get('query')!=query_digest or type(payload.get('offset')) is not int or payload['offset']<0:
                raise RetrievalError('knowledge cursor does not match this package or search')
            offset=payload['offset']
        allowed=None
        if topic is not None:
            allowed={'card:'+card_id for card_id in self.packs[topic]['card_ids']}
        q=set(_tokens(combined)); scored=[]
        for item in self.index.get('items',[]):
            if allowed is not None and item['id'] not in allowed:
                continue
            if not combined and topic is not None:
                score=0
            else:
                score=(5*len(q & set(_tokens(item.get('title','')+' '+' '.join(item.get('aliases',[])))))
                       +3*len(q & set(_tokens(item.get('summary',''))))
                       +2*len(q & set(_tokens(' '.join(item.get('source_features',[])))))
                       +len(q & set(_tokens(item.get('text','')))))
            if score or (topic is not None and not combined): scored.append((-score,item['id'],item))
        matches=[row[2] for row in sorted(scored)]
        if offset>len(matches):
            raise RetrievalError('knowledge cursor is outside the result set')
        page=matches[offset:offset+limit]
        next_offset=offset+len(page)
        next_cursor=(self._cursor({'kind':'search','package':self.package_digest,
                                    'query':query_digest,'offset':next_offset})
                     if next_offset<len(matches) else None)
        return {'items':page,'next_cursor':next_cursor,'total_matches':len(matches),'topics':topics}
    def _load(self,ref):
        kind,sep,name=ref.partition(':')
        if not sep or kind not in ('card','ref') or '/' in name or '..' in name: raise RetrievalError('invalid semantic reference')
        obj=(self.cards if kind=='card' else self.refs).get(name)
        if obj is None: raise RetrievalError('unknown semantic reference')
        return kind,name,obj
    def read(self,ref,max_bytes=12000,offset=0,*,paginate=False):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise RetrievalError('invalid knowledge read budget')
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise RetrievalError('invalid knowledge read offset')
        if offset != 0 and not paginate:
            raise RetrievalError('knowledge reads require offset zero')
        kind,name,obj=self._load(ref); closure=[]; seen=set()
        def visit(r):
            if r in seen:return
            seen.add(r); k,n,o=self._load(r); closure.append((r,o))
            for dep in o.get('requires',[]): visit(dep)
        visit(ref); content=json.dumps({'knowledge':[o for _,o in closure]},ensure_ascii=False,sort_keys=True,indent=2)+'\n'; size=len(content.encode())
        if paginate:
            if kind != 'ref':
                raise RetrievalError('explicit pagination is available only for references')
            raw = obj['body'].encode('utf-8')
            if offset > len(raw):
                raise RetrievalError('reference offset is outside its body')
            try:
                remaining = raw[offset:].decode('utf-8')
            except UnicodeDecodeError as exc:
                raise RetrievalError('reference offset must be a UTF-8 boundary') from exc
            def page(length):
                fragment = remaining[:length]
                end = offset + len(fragment.encode('utf-8'))
                root = dict(obj, body=fragment)
                text = json.dumps({'knowledge': [root] + [o for _,o in closure[1:]],
                                   'reference_page': {'ref': ref, 'body_offset': offset,
                                                      'next_offset': end if end < len(raw) else None,
                                                      'body_total_bytes': len(raw)}},
                                  ensure_ascii=False, sort_keys=True, indent=2) + '\n'
                return text, end
            # The complete applicability and dependencies repeat on every page;
            # a tiny budget never returns an incomplete prerequisite or JSON object.
            minimum, _ = page(min(1, len(remaining)))
            if len(minimum.encode()) > max_bytes:
                return {'ref':ref,'content':'','offset':offset,'truncated':True,
                        'needs_more_budget':True,'required_bytes':len(minimum.encode())}
            low, high = min(1, len(remaining)), len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if len(page(middle)[0].encode()) <= max_bytes:
                    low = middle
                else:
                    high = middle - 1
            content, end = page(low)
            return {'ref':ref,'content':content,'offset':offset,'truncated':end < len(raw),
                    'needs_more_budget':False,'required_bytes':size}
        if size>max_bytes:return {'ref':ref,'content':'','offset':offset,'truncated':True,'needs_more_budget':True,'required_bytes':size}
        return {'ref':ref,'content':content,'offset':offset,'truncated':False,'needs_more_budget':False,'required_bytes':size}

    def read_page(self,ref,max_bytes=12000,*,cursor=None):
        """Return one complete JSON page from a root object and its requires closure."""
        if type(max_bytes) is not int or not 1<=max_bytes<=32768:
            raise RetrievalError('invalid knowledge read page budget')
        _,_,root=self._load(ref)
        closure=[]; seen=set()
        def visit(current):
            if current in seen:return
            seen.add(current)
            _,_,value=self._load(current)
            closure.append((current,value))
            for dependency in sorted(value.get('requires',[])):
                visit(dependency)
        visit(ref)
        position=0; body_offset=0
        include_dependencies=cursor is None
        if cursor is not None:
            payload=self._decode_cursor(cursor)
            if (payload.get('kind')!='read' or payload.get('package')!=self.package_digest
                    or payload.get('ref')!=ref or type(payload.get('position')) is not int
                    or type(payload.get('body_offset')) is not int):
                raise RetrievalError('knowledge cursor does not match this package or reference')
            position=payload['position']; body_offset=payload['body_offset']
            if position<0 or position>=len(closure) or body_offset<0:
                raise RetrievalError('knowledge cursor is outside the dependency closure')
        raw_page=[]
        dependency_refs=[item_ref for item_ref,_ in closure[1:]]
        root_requires=list(root.get('requires',[]))
        root_related=list(root.get('related',[])) if ref.startswith('card:') else []

        def cursor_for(pos, off):
            if pos>=len(closure): return None
            return self._cursor({'kind':'read','package':self.package_digest,'ref':ref,
                                 'position':pos,'body_offset':off})

        def encode(entries, next_cursor):
            content=json.dumps({'knowledge':entries},ensure_ascii=False,sort_keys=True,separators=(',',':'))
            result={'schema_version':1,'ref':ref,'content':content,'offset':body_offset,
                    'truncated':next_cursor is not None,'needs_more_budget':False,
                    'required_bytes':None,'requires':root_requires,'related':root_related,
                    'dependencies':dependency_refs if include_dependencies else [],'next_cursor':next_cursor,
                    'complete':next_cursor is None}
            return content,result,len(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode('utf-8'))

        pos=position; off=body_offset
        while pos<len(closure):
            current,value=closure[pos]
            kind=current.partition(':')[0]
            if kind=='ref':
                body=value['body']; body_bytes=body.encode('utf-8')
                if off>len(body_bytes):
                    raise RetrievalError('knowledge cursor body offset is invalid')
                try: remaining=body_bytes[off:].decode('utf-8')
                except UnicodeDecodeError as exc: raise RetrievalError('knowledge cursor is not on a UTF-8 boundary') from exc
                # If the complete reference and the remaining closure fit, keep it whole.
                next_pos=pos+1
                if off == 0:
                    candidate=dict(value)
                    entry={'ref':current,'value':candidate}
                else:
                    candidate=dict(value,body=remaining)
                    entry={'ref':current,'value':candidate,
                           'body_page':{'offset':off,'next_offset':None,
                                        'total_bytes':len(body_bytes)}}
                next_cursor=cursor_for(next_pos,0)
                candidate_page,_,size=encode(raw_page+[entry],next_cursor)
                if size<=max_bytes:
                    raw_page.append(entry); pos=next_pos; off=0
                    continue
                # A long reference is paged by UTF-8 code point. Its semantic
                # fields are repeated, but dependency objects are never repeated.
                low,high=1,len(remaining)
                best=None
                while low<=high:
                    middle=(low+high)//2
                    fragment=remaining[:middle]
                    end=off+len(fragment.encode('utf-8'))
                    next_state=(pos,end) if end<len(body_bytes) else (pos+1,0)
                    page_cursor=cursor_for(*next_state)
                    partial=dict(value,body=fragment)
                    entry={'ref':current,'value':partial,
                           'body_page':{'offset':off,'next_offset':end if end<len(body_bytes) else None,
                                        'total_bytes':len(body_bytes)}}
                    page_text,_,candidate_size=encode(raw_page+[entry],page_cursor)
                    if candidate_size<=max_bytes:
                        best=(entry,next_state,page_cursor,page_text)
                        low=middle+1
                    else:
                        high=middle-1
                if best is None:
                    if raw_page:
                        break
                    raise RetrievalError('read page budget is too small for required reference metadata; use max_bytes=32768')
                raw_page.append(best[0]); pos,off=best[1]
                # A partial reference must end the page so a caller can consume
                # its body fragments in order before following dependencies.
                break
            entry={'ref':current,'value':value}
            next_cursor=cursor_for(pos+1,0)
            _,_,size=encode(raw_page+[entry],next_cursor)
            if size>max_bytes:
                if raw_page: break
                raise RetrievalError('read page budget is too small for the root or dependency item; use max_bytes=32768')
            raw_page.append(entry); pos+=1; off=0

        next_cursor=cursor_for(pos,off)
        content,result,size=encode(raw_page,next_cursor)
        if size>max_bytes:
            raise RetrievalError('read page envelope exceeds its byte budget')
        result['content']=content
        return result


def run_retrieval_qa(root, topics):
    """Run adversarial and configured-question retrieval QA against a staging package."""
    store = KnowledgeStore(root)
    rows = []
    for topic in topics:
        topic_id = topic['id']
        title = topic.get('title', topic_id)
        queries = (
            ('normal', f'{title} principles conditions'),
            ('wrong_premise', f'{title} always fastest without conditions'),
            ('insufficient', f'{title} unknown API evidence'),
        )
        for kind, query in queries:
            hits = store.search(query, limit=5)
            rows.append({'topic': topic_id, 'kind': kind, 'query': query, 'hit_ids': [x['id'] for x in hits], 'ok': any(x['id'] == f'card:{topic_id}' for x in hits)})
        for number, question in enumerate(topic.get('questions', []), start=1):
            query = question
            hits = store.search(query, limit=5)
            rows.append({'topic': topic_id, 'kind': 'required_question',
                         'question_id': f'{topic_id}.q{number:02d}', 'question': question,
                         'query': query, 'hit_ids': [x['id'] for x in hits],
                         'ok': any(x['id'] == f'card:{topic_id}' for x in hits)})
    # Use nonce-only probes so a legitimate card containing a common English
    # word cannot make the unrelated-query gate fail.
    for query in ('zqxv qjrm', 'kxpt vbnm', 'mnrq tpsd', 'hkwp yxzr', 'qfjd plmk'):
        hits = store.search(query, limit=5)
        rows.append({'topic': None, 'kind': 'unrelated', 'query': query, 'hit_ids': [x['id'] for x in hits], 'ok': not hits})
    return {'ok': all(x['ok'] for x in rows), 'rows': rows}
