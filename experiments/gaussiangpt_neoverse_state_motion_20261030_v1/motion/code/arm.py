"""A bounded independent motion arm: real backward witness -> train -> evaluate."""
import argparse,subprocess,sys,traceback
from pathlib import Path
from motion_common import read,write
def main(root,arm):
    folder=root/arm;folder.mkdir(parents=True,exist_ok=True);phase='start'
    try:
        for phase,script,limit in [('witness','witness_motion.py',900),('train','train_motion.py',15000),('eval','evaluate_motion.py',9000)]:
            write(folder/'arm_status.json',{'status':'running','phase':phase})
            process=subprocess.Popen([sys.executable,'-u','-B',str(root/'code'/script),'--root',str(root),'--arm',arm])
            try:
                code=process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                # Give the training handler time to save model/optimizer/RNG.
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=20)
                raise
            if code:raise RuntimeError(f'{phase} exited with code{code}')
            if phase=='train':
                t=read(folder/'train_done.json');assert t['steps']==3000 and t['all_pairs_seen'] and t['no_target_forcing_render_steps']>0
        write(folder/'arm_status.json',{'status':'complete','phase':'finished'})
    except BaseException as e:
        write(folder/'arm_failure.json',{'status':'failed','phase':phase,'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();main(a.root,a.arm)
