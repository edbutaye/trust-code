import unittest
from mpi4py import MPI

class trusticoco_test(unittest.TestCase):

    def test_set_comm(self):
        return
        """ Set a MPI communicator created from Python """
        import trusticoco as ti

        pbT = ti.ProblemTrio()
        pbT.name = "TRUST"
        com2 = MPI.Comm(MPI.COMM_WORLD)
        pbT.setMPIComm(com2)

    def test_repeated_run_parallel(self):
        """ Same as test_repeated_run() (in  test_trusticoco.py) but in //
        """
        import trusticoco as ti
        import medcoupling as mc

        # ProblemTrio should be instanciated only once ! 
        # But after that, initialize() and terminate() may be called several times
        pbT = ti.ProblemTrio()
        pbT.name = "TRUST"
        com2 = MPI.Comm(MPI.COMM_WORLD)
        pbT.setMPIComm(com2)

        def run():
            pbT.setDataFile("PAR_test_conduc_para.data")
            pbT.initialize()
            dt, stop = pbT.computeTimeStep()
            pbT.initTimeStep(dt)
            ok = pbT.solveTimeStep()
            pbT.validateTimeStep()
            pbT.terminate()

        run()
        run()
        run()

if __name__ == "__main__":
    unittest.main()
