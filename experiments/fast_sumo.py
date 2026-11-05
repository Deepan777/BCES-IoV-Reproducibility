"""Identical SUMO command/control, shorter client connection retry interval.

The standard TraCI launcher waits one second after an initial refused connect.
This launcher changes only that wall-clock retry delay, not simulated time,
commands, RNG seeds, network, vehicle controls, sensing, or planner behavior.
"""
import contextlib
import io
from pathlib import Path
import socket
import subprocess
import sys

sys.dont_write_bytecode=True
ORIGINAL=Path(r'D:\VIT Vellore Research one\BCES-IoV')
sys.path.insert(0,str(ORIGINAL))
from bces.simulation.sumo_adapter import sumo_binary,load_traci

def start_controlled(*,network,label,seed,step_s,state=None):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1',0));port=server.getsockname()[1]
    command=[str(sumo_binary()),'-n',str(network),'--step-length',str(step_s),'--seed',str(seed),
        '--no-step-log','true','--no-warnings','true','--time-to-teleport','-1',
        '--collision.action','warn','--save-state.rng','true','--save-state.transportables','true',
        '--save-state.precision','12','--thread-rngs','1','--remote-port',str(port)]
    if state is not None:command.extend(['--load-state',str(state)])
    process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            connection=load_traci().connect(port,numRetries=250,host='127.0.0.1',proc=process,waitBetweenRetries=.02)
        connection.getVersion()
        return connection
    except Exception:
        if process.poll() is None:process.terminate()
        _,err=process.communicate(timeout=5)
        raise RuntimeError('SUMO startup failed: '+err.decode(errors='replace'))

def validate():
    import hashlib,json,time
    from dataclasses import asdict
    import yaml
    from development_study import OUT,save_json
    from bces.simulation.controlled_slots import development_generator
    generator=development_generator()
    original=generator.start_controlled
    config=yaml.safe_load((ORIGINAL/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    config['ages_s']=[.4]
    results=[]
    for seed in (1600000000,1600000002):
        spec,params=generator.sample_scenario(seed,config)
        start=time.perf_counter();old=generator.collect_trace(seed,spec,params,config,3.);oldtime=time.perf_counter()-start
        generator.start_controlled=start_controlled
        try:
            start=time.perf_counter();new=generator.collect_trace(seed,spec,params,config,3.);newtime=time.perf_counter()-start
        finally:generator.start_controlled=original
        def encode(trace):return json.dumps({t:{'receiver':f['receiver'],'objects':[asdict(o) for o in f['objects']]} for t,f in trace.items()},sort_keys=True,allow_nan=False).encode()
        a,b=encode(old),encode(new);assert a==b,'fast connection changed the recorded simulated world'
        results.append({'seed':seed,'identical_receiver_and_world_frames':True,'frames':len(old),
            'trace_sha256':hashlib.sha256(a).hexdigest(),'standard_wall_s':oldtime,'fast_wall_s':newtime})
    save_json(OUT/'fast_sumo_replay_validation.json',{'changes_only_wall_clock_retry':True,'results':results})
    print(json.dumps(results),flush=True)

if __name__=='__main__':validate()
