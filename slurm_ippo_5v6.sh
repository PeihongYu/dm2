#!/bin/bash
#SBATCH --job-name 5v6_dm2-qmix   			                                        # Job name
### Logging
#SBATCH --output=/scratch/cluster/clw4542/marl_results/ippo_5v6/5v6_%A_%a.out           # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/clw4542/marl_results/ippo_5v6/5v6_%A_%a.err            # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job
### Node info
#SBATCH --partition titans                                                      # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 200:00:00                                                      # Run time (hh:mm:ss)
#SBATCH --gres=gpu:1                                                         # Number of gpus needed
#SBATCH --mem=40G                                                            # Memory requirements
#SBATCH --cpus-per-task=8                                                    # Number of cpus needed per task

# seedarray=(112358 1285842 78590 119527 122529)
# sleep $SLURM_ARRAY_TASK_ID
# seed=${seedarray[SLURM_ARRAY_TASK_ID]}
python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=112358 &
python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=1285842 &
# python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=78590 &
# python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=119527 &
# python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=122529 &

wait
# done
echo "Done"
exit 0