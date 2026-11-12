import sys,unittest,math,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dtr import *
class Math(unittest.TestCase):
    def test_jsd(self):
        a=torch.tensor([5.,0.,-5.]).log_softmax(-1);b=-a;b=b.log_softmax(-1)
        self.assertLess(float(jsd(a,a)),1e-7);self.assertAlmostEqual(float(jsd(a,b)),float(jsd(b,a)),places=6);self.assertTrue(0<=float(jsd(a,b))<=math.log(2))
    def test_crossing(self):
        a=[.6]*23+[.4]*4+[0.];b=[.6]*22+[.4]*5+[0.]
        self.assertEqual(depths([a])['first_crossing'],[24]);self.assertEqual(depths([a])['dtr'],1);self.assertEqual(depths([b])['dtr'],0)
    def test_rebound(self):
        row=[.1]+[.6]*26+[0.];r=depths([row]);self.assertEqual(r['first_crossing'],[1]);self.assertEqual(r['persistent'],[28])
    def test_empty(self):
        with self.assertRaises(ValueError):depths([])
    def test_certainty_direction(self):
        lp=torch.tensor([4.,1.,-5.]).log_softmax(-1)
        self.assertAlmostEqual(float(certainty(lp)),float((torch.ones(3)/3*(-math.log(3)-lp)).sum()),places=6)
        self.assertNotAlmostEqual(float(certainty(lp)),float((lp.exp()*(lp+math.log(3))).sum()),places=3)
    def test_full_vocab(self):
        a=torch.tensor([.8,.1,.1]);b=torch.tensor([.8,.19,.01]);self.assertGreater(float(jsd(a.log(),b.log())),0)
        self.assertEqual(float(jsd(torch.tensor([0.]),torch.tensor([0.]))),0.)
    def test_bits_degenerate(self):
        self.assertAlmostEqual(math.ceil(.85*28),24);self.assertEqual(depths([[math.log(2)]*27+[0]],g=.75)['first_crossing'],[1])
if __name__=='__main__':unittest.main()
