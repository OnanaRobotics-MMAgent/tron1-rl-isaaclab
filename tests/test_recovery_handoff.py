import sys
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/rsl_rl'))
from recovery_handoff import WheelLegFK, HandoffController, handoff_conditions, blend_actions


class TestHandoff(unittest.TestCase):
    def test_nominal_pose_ready_but_inverted_and_airborne_rejected(self):
        names = [f'{k}_{s}_Joint' for k in ('abad','hip','knee','wheel') for s in ('L','R')]
        fk = WheelLegFK(ROOT/'exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/WF_TRON1A.urdf', names)
        q = torch.tensor([[0.,0.,.13437526,-.13437526,-.53190465,.53190465,0.,0.]])
        g = torch.tensor([[0.,0.,-1.]])
        geom = fk.compute(q, g)
        args = (geom['lengths'], torch.zeros(1,3), torch.ones(1,2)*20)
        self.assertTrue(handoff_conditions(geom,*args)[0].item())
        self.assertFalse(handoff_conditions(fk.compute(q,-g),*args)[0].item())
        self.assertFalse(handoff_conditions(geom,args[0],args[1],torch.zeros(1,2))[0].item())
        torch.testing.assert_close(geom['support_heights'],torch.tensor([[.18,.18]]),atol=.001,rtol=0)
        # Continuous wheel revolution cannot change cylinder center/height.
        q[:,6:] = 23.1
        torch.testing.assert_close(fk.compute(q,g)['centers'],geom['centers'])

    def test_debounce_blend_ramp_and_fallback(self):
        c = HandoffController(1,.02,hold=.1,blend=.2)
        yes,no = torch.tensor([True]),torch.tensor([False])
        for _ in range(4):
            c.update(yes,no)
        self.assertEqual(c.phase.item(),0)
        c.update(no,no)
        self.assertEqual(c.ready_time.item(),0)
        for _ in range(5):
            c.update(yes,no)
        self.assertEqual(c.phase.item(),1)
        self.assertEqual(c.walk_time.item(),0)
        for _ in range(80):
            a,cmd=c.update(yes,no)
        self.assertEqual(c.phase.item(),2)
        self.assertEqual(a.item(),1)
        self.assertEqual(cmd.item(),1)
        for _ in range(8):
            a,_=c.update(no,yes)
        self.assertEqual(c.phase.item(),3)
        self.assertGreater(a.item(),0)
        for _ in range(30):
            c.update(no,yes)
        self.assertEqual(c.phase.item(),0)
        c.reset(yes)
        self.assertEqual(c.progress.item(),0)

    def test_physical_target_slew_and_clipping(self):
        rec,loc=torch.tensor([[20.,-10.]]),torch.tensor([[-20.,10.]])
        scale,offset=torch.tensor([.12,53.333]),torch.tensor([.1,0.])
        low,high=torch.tensor([-1.,-60.]),torch.tensor([1.,60.])
        prev=torch.zeros(1,2)
        rate=torch.tensor([3.,80.])
        action=blend_actions(rec,loc,torch.tensor([.2]),prev,scale,offset,low,high,rate,.02,torch.tensor([True]))
        self.assertTrue(((action-prev).abs()*scale <= rate*.02+1e-6).all())
        untouched=blend_actions(rec,loc,torch.tensor([0.]),prev,scale,offset,low,high,rate,.02,torch.tensor([False]))
        torch.testing.assert_close(untouched,rec)


if __name__=='__main__':
    unittest.main()
