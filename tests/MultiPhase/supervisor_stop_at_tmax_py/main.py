# -*- coding: utf-8 -*-


class TRUSTICoCo():
    """
    TRUSTICoCoclass which implements the RunICoCo method
    """
    @classmethod
    def RunICoCo(self):
        """
        Execute ...
        """
        import medcoupling as mc
        import trusticoco as ti
        import time

        print ("Using ICoCo version",ti.ICOCO_VERSION)
        pbT = ti.ProblemTrio()
        pbT.name = "TRUST"
        pbT.setDataFile("jdd.data")
        pbT.initialize()

        porosity = mc.ICoCoMEDDoubleField()

        print("InputFieldsNames : ",pbT.getInputFieldsNames())
        print("OutputFieldsNames : ",pbT.getOutputFieldsNames())

        def run(pb):
            """
            Internal method for RunICoCo. It defines the time looping scheme.
            """
            stop = False # Does the Problem want to stop ?
            t = 0.0

            init = True
            while not stop:
                dt,stop = pbT.computeTimeStep()
                if stop: 
                    return

                pbT.initTimeStep(dt)
                
                t = pbT.presentTime() # time av_totalanced

                # Main time loop:
                ok = pbT.solveTimeStep()
                t = pbT.presentTime() # time av_totalanced

                # iterate or stop ?
                if (not ok): # The resolution failed, try with a new time interval.
                    pbT.abortTimeStep()
                else: # The resolution was successful, validate and go to the next time step.
                    pbT.validateTimeStep()

                init = False
                print("=================================================================")

            stat = pbT.isStationary()
            if (stat):
                stop = True

        run(pbT)
        pbT.terminate()

if __name__ == "__main__":
    TRUSTICoCo().RunICoCo()
