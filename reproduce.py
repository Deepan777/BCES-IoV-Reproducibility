"""Materialize generated experimental data and replay fixed analyses."""
import argparse,gzip,hashlib,importlib.util,json,os,runpy,shutil,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def configure():
 sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
 import bces
 assert Path(bces.__file__).resolve().is_relative_to(ROOT),'Implementation imported outside checkout'
 import development_study
 old=str(development_study.ORIGINAL);development_study.ORIGINAL=ROOT
 import fast_sumo
 fast_sumo.ORIGINAL=ROOT;sys.path[:]=[p for p in sys.path if p!=old]+[str(ROOT)]
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def materialize():
 count=0
 for pack in load(ROOT/'data/reproduction_manifest.json')['data_packs']:
  p=ROOT/pack['path'];assert digest(p)==pack['sha256'];expected={x['path']:x for x in pack['files']};seen=set()
  with tarfile.open(p,'r|gz') as t:
   for x in t:
    if not x.isfile() or x.name not in expected:raise ValueError('Unexpected archive member')
    p=(ROOT/x.name).resolve();assert p.is_relative_to(ROOT);info=expected[x.name];seen.add(x.name)
    if p.is_file():assert digest(p)==info['sha256'],'Existing data changed: '+str(p)
    else:
     b=t.extractfile(x).read();assert len(b)==info['bytes'] and hashlib.sha256(b).hexdigest()==info['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    count+=1
  assert seen==set(expected);print(json.dumps({'materialized':pack['path'],'files':len(seen)}),flush=True)
 print(json.dumps({'files':count}),flush=True)
def verify(expanded=False):
 m=load(ROOT/'data/reproduction_manifest.json');n=0
 for p,x in m.get('direct_files',{}).items():assert digest(ROOT/p)==x['sha256'];n+=1
 for pack in m['data_packs']:
  assert digest(ROOT/pack['path'])==pack['sha256']
  if expanded:
   for x in pack['files']:assert digest(ROOT/x['path'])==x['sha256'];n+=1
 if expanded:
  configure();from confirmation_study import verify_registration
  verify_registration(load(ROOT/'experiments/confirmation_registration.json'))
 print(json.dumps({'integrity':'PASS','files':n,'data_packs':len(m['data_packs']),'registered_inputs_verified':expanded}),flush=True)
def primary():
 from bces.evaluation.slot_confirmation import summarize
 reg=load(ROOT/'outputs/study_b/slot_confirmation_v1/preregistration.json')
 for phase in ['calibration','confirmation']:
  folder=ROOT/'outputs/study_b/slot_confirmation_v1'/phase;m=load(folder/'run_manifest.json');rows=[]
  for n,h in m['artifact_sha256'].items():
   assert digest(folder/n)==h
   with gzip.open(folder/n,'rt',encoding='utf-8') as f:r=json.load(f)
   assert r['registration_sha256']==digest(ROOT/'outputs/study_b/slot_confirmation_v1/preregistration.json');rows.append(r['evaluation'])
  result=summarize(rows,expected_n=reg['plan'][phase]['scenarios'])
  assert result==m['results'],'Primary analysis mismatch';save(ROOT/'reproduced'/('primary_'+phase+'.json'),result);print(json.dumps({'study':'primary_'+phase,'matches':True,'scenarios':len(rows)}),flush=True)
def broad():module(ROOT/'operations/verify_completed_evidence.py','verify_cases').verify_slots()
def driving():
 module(ROOT/'operations/verify_completed_evidence.py','verify_cases').verify_driving()
 from driving_study import summarize,paired_analysis
 reg=load(ROOT/'experiments/confirmation_registration.json');folder=ROOT/'experiments/results/driving_confirmation';rows=[]
 for a in reg['driving_assignments']:
  for c in reg['driving_conditions']:
   for m in reg['driving_methods']:
    with gzip.open(folder/f'{a["seed"]}_{c}_{m}.json.gz','rt',encoding='utf-8') as f:rows.append(json.load(f))
 ref=load(folder/'report.json');s=summarize(rows);p=paired_analysis(rows,reg)
 assert s==ref['summaries'],'Driving summary mismatch';assert p==ref['paired_primary_planning_score'],'Driving inference mismatch'
 save(ROOT/'reproduced/driving.json',{'summaries':s,'paired_primary_planning_score':p});print(json.dumps({'study':'receiver_local_driving','matches':True,'branches':len(rows)}),flush=True)
def source_query():
 m=module(ROOT/'scripts/32_analyze_closed_loop_v2.py','source_query');original=m.OUTPUT;target=ROOT/'reproduced/source_query';target.mkdir(parents=True,exist_ok=True)
 for p in original.iterdir():
  if p.name=='run_manifest.json' or p.name.startswith('branch_') and p.name.endswith('.json.gz'):
   q=target/p.name
   if not q.exists():
    try:os.link(p,q)
    except OSError:shutil.copy2(p,q)
 m.OUTPUT=target;m.main();a=load(target/'secondary_analysis.json');b=load(original/'secondary_analysis.json')
 for k in ['completed_utc','generated_utc']:a.pop(k,None);b.pop(k,None)
 assert a==b,'Source-query analysis mismatch';print(json.dumps({'study':'source_query','matches':True}),flush=True)
def robustness():
 from scipy.stats import beta
 folder=ROOT/'experiments/results/fresh_robustness';rows=load(folder/'rows.json');ref=load(folder/'report.json');out={}
 for c in ref['methods']:
  values=[r for r in rows if r['condition']==c];out[c]={}
  for m in ref['methods'][c]:
   a=sum(r['accepted'][m] for r in values);k=sum(r['accepted'][m] and r['valid'] is not True for r in values)
   out[c][m]={'scenarios':len(values),'accepted':a,'coverage':a/len(values),'invalid_or_unknown_accepted':k,'invalid_reuse':k/a if a else None,'family_adjusted_upper95':float(beta.ppf(1-.05/(len(ref['methods'])*len(ref['methods'][c])),k+1,a-k)) if k<a else 1.}
 assert out==ref['methods'],'Sensing-error analysis mismatch';save(ROOT/'reproduced/sensing_errors.json',out);print(json.dumps({'study':'sensing_errors','matches':True}),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('materialize');a=sub.add_parser('verify');a.add_argument('--materialized',action='store_true');a=sub.add_parser('analyze');a.add_argument('study',choices=['primary','broad','driving','source-query','robustness','support','all']);a=sub.add_parser('run');a.add_argument('script');a.add_argument('args',nargs=argparse.REMAINDER);args=p.parse_args()
 if args.command=='materialize':materialize()
 elif args.command=='verify':verify(args.materialized)
 else:
  configure()
  if args.command=='run':
   f=(ROOT/args.script).resolve();assert f.is_relative_to(ROOT) and f.suffix=='.py';sys.argv=[str(f),*args.args];runpy.run_path(str(f),run_name='__main__')
  else:
   tasks={'primary':primary,'broad':broad,'driving':driving,'source-query':source_query,'robustness':robustness,'support':lambda:module(ROOT/'operations/reproduce_support.py','support_diagnostics').main()}
   for n in (tasks if args.study=='all' else [args.study]):tasks[n]()
if __name__=='__main__':main()
