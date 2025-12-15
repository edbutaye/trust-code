#!/bin/bash
##################################
# Variables for configure script #
##################################
define_modules_config()
{
   env=$TRUST_ROOT/env/machine.env
   # Initialisation de l environnement module $MODULE_PATH si pas disponible:
   module -v 1>/dev/null 2>&1 || echo $echo "source /etc/profile" >> $env
   #
   # Load modules
   if [ "$TRUST_USE_CUDA" = 1 ]
   then
      TRUST_CUDA_CC=90
      module="bsc/1.0 nvidia-hpc-sdk/25.1 cmake/3.29.2 ucx/1.16.0 openmpi/4.1.5-ucx1.16-gcc "
   else
      echo "Not supported yet"
      exit 1
   fi
   #
   echo "# Module $module detected and loaded on $HOST."
   echo "module purge 1>/dev/null" >> $env
   echo "module load $module 1>/dev/null || exit -1" >> $env
   . $env
   # Creation wrapper qstat -> squeue
   echo "#!/bin/bash
squeue" > $TRUST_ROOT/bin/qstat
   chmod +x $TRUST_ROOT/bin/qstat
}

##############################
# Variables for trust script #
##############################
define_soumission_batch()
{
   soumission=1
   # https://www.bsc.es/supportkc/docs/MareNostrum5/slurm
   if [ "$gpu" = 1 ]
   then
      ntasks=80 # number of cores max
      if [ $NB_PROCS -le 4 ]
      then
         gpus_per_node=$NB_PROCS
      else
         gpus_per_node=4
	 node=1 # --exclusive
      fi	 
      cpus_per_task=20
      # 400 GPUs max
      qos=acc_ehpc && cpu=4320 && [ "$prod" != 1 ] && [ $NB_PROCS -le 32 ] && qos=acc_debug && cpu=120 
      project="ehpc469"
   else
      echo "Not supported yet." && exit 1
   fi
   mpirun="srun -n \$SLURM_NTASKS"
   sub=SLURM
}

