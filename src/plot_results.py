import os
os.chdir("..")
print(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import glob
import copy

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

from utils.vis_utils import combine_tb_logs



def gen_log_data(experiments:dict):
    exp_dfs = {}
    for exp_name, exp_files_list in experiments.items():
        if "ippo" in exp_name.lower(): 
            stat_name = "test_ippo_battle_won_mean"
        elif "dm2" in exp_name.lower():
            stat_name = "test_ippo_battle_won_mean"            
        elif "qmix" in exp_name.lower(): 
            stat_name = "test_qmix_battle_won_mean"
        elif "rmappo" in exp_name.lower():
            stat_name = "eval_win_rate"
        else:
            stat_name = "test_ippo_battle_won_mean"

        res_dict = combine_tb_logs(exp_files_list, stat_name, ts_round_base=5e+4)
        
        # trim logs
        log_lens = []
        for k, v in res_dict.items():
            log_lens.append(len(v))
        min_log_len = min(log_lens)

        for k, v in res_dict.items():
            res_dict[k] = v[:min_log_len]
        
        # convert to dataframe
        df = pd.DataFrame.from_dict(res_dict, orient='columns')
        # breakpoint()
        df = df.melt(id_vars="ts", var_name="runs", value_name="test_battle_won_mean")
        # breakpoint()
        
        exp_dfs[exp_name] = df
        # breakpoint()

    return exp_dfs

# def core_plots(exp_dfs:dict, demo_qual:dict, )
def plot_experiments(exp_dfs:dict, 
                     savename:str, 
                     plot_title:str=None, 
                     legend=True,
                     legend_cols=3,
                     legend_loc=(1.0, -0.25),
                     demo_qual:dict=None, 
                     yaxis_lims=None,
                     custom_color_order=None,
                     save=False, 
                     savedir="figures/"):
#     sns.set_theme(style="wgrid")
    sns.set_context("paper")
    sns.set(font_scale = 1.5)
    
    if custom_color_order is None:
        custom_color_order = list(range(len(exp_dfs)))

    baseline_palette = list(sns.color_palette("mako", as_cmap=False, n_colors=7).as_hex())
    # dm2_palette = list(sns.color_palette("mako", as_cmap=False, n_colors=12).as_hex())
    dm2_palette = list(sns.color_palette("Oranges", as_cmap=False, n_colors=7).as_hex())
    # gail_palette = list(sns.color_palette("BuPu", as_cmap=False, n_colors=7, desat=0.7).as_hex())
    # dm2_palette = list(sns.color_palette("Greens", as_cmap=False, n_colors=8).as_hex())
    gail_palette = list(sns.color_palette("Blues", as_cmap=False, n_colors=8).as_hex())
    # gail_palette = list(sns.color_palette("tab10", as_cmap=False, n_colors=8).as_hex())[::-1]
    # gail_palette = sns.color_palette("colorblind", as_cmap=True)
    default_palette = sns.color_palette("colorblind", as_cmap=True)
    dm2_palette.pop()
    # dm2_palette.pop()
    gail_palette.pop()
    default_palette = sns.color_palette("Paired").as_hex()

    # sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_style("ticks")
    figure, axis = plt.subplots(1, 1, figsize=(7, 4.3)) # figsize argument
    for i, (exp_name, exp_df) in enumerate(exp_dfs.items()):
        # breakpoint()
        if "IPPO" in exp_name: 
            # color = baseline_palette.pop()
            # baseline_palette.pop()
            color = default_palette[3]
        elif "DM2" in exp_name: 
            # color = dm2_palette[i*2 + 3]
            color = dm2_palette.pop()
            dm2_palette.pop()
            # if 'diff' in exp_name:
            #     color = default_palette[2]
            # else:
            #     color = default_palette[3]
        elif "PegMARL" in exp_name:
            color = gail_palette.pop()
            gail_palette.pop()
            # if 'diff' in exp_name:
            #     color = default_palette[0]
            # else:
            #     color = default_palette[1]
        else: 
            color = default_palette[i]
        sns.lineplot(data = exp_df, x="ts", y="test_battle_won_mean", 
                     ci="sd", ax=axis, label=exp_name, 
                     color=color,
                    )
        
    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Mean Battle Won Rate")
    if yaxis_lims is not None:
        axis.set_ylim(yaxis_lims)
    if plot_title is not None:
        axis.set_title(plot_title, style='italic')

    if demo_qual is not None:
        color_list = ["darkslategray", "slateblue", "darkslateblue", "indigo", "darkviolet"]
        # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        linestyle_list = [
             ('densely dashed',        (0, (5, 1))),
             ('loosely dotted',        (0, (1, 10))),
             ('dotted',                (0, (1, 1))),
             ('loosely dashed',        (0, (5, 10))),
             ('long dash with offset', (5, (10, 3))),
              ]
        for i, (demo_name, demo_quality) in enumerate(demo_qual.items()):
            plt.axhline(y=demo_quality, xmin=0.05, xmax=0.95, 
                        color=color_list[0], # figure_colors[i+1], 
                        linestyle= linestyle_list.pop(-1)[1], 
                        label=demo_name
                       )
    if legend:
        legend_loc=(0.45, 1.3)
        plt.legend(loc='upper center',
                   bbox_to_anchor=legend_loc, 
                   borderaxespad=0.,
                   ncol=legend_cols,
                   fontsize=14
                  )
    else:
        leg = axis.get_legend()
        leg.remove()
    
#     plt.tight_layout()
    if save:
        # if not os.path.exists(savedir):
        #     os.mkdir("figures")            
        savepath = f"/fs/nexus-projects/Guided_MARL/peihong_dm2_results/{savename}.pdf"
        
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


# append base paths
    


base_paths = {"ippo": "/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/{}/"}

demo_win_rate = {
    "5m_vs_6m": {
#         "ippo demo (4m)": 0.015748031437397003,
#         "ippo demo (5m)": 0.12598425149917603, # NEW 
        # "ippo demo (5m)": 0.30952380952380953,# OLD
        "demo": 0.30952380952380953,# OLD
#         "ippo demo (6m)": 0.22047244012355804,
#         "ippo demo (7m)": 0.4409448802471161, # NEW
        # "ippo demo (7m)": 0.48412698412698413,# OLD
#         "ippo demo (8m)": 0.4094488322734833,
#         "ippo demo (9m)": 0.5196850299835205, 
#         "dippo demo (10m)": 0.5826771855354309,
           },
    "3s_vs_4z": {
#         "ippo demo (4m)": 0.06299212574958801,
#         "ippo demo (5m)": 0.3385826647281647, # NEW
        # "ippo demo (5m)": 0.35714285714285715, # OLD
        "demo": 0.35714285714285715, # OLD
#         "ippo demo (6m)": 0.5118110179901123, # NEW
        # "ippo demo (6m)":0.5634920634920635, # OLD
#         "ippo demo (7m)": 0.6535432934761047,
#         "ippo demo (8m)": 0.6771653294563293,
#         "ippo demo (9m)": 0.8110235929489136, 
    },
}

# # construct baselines expt dict
# # dict structure: algo name, task name, experiment name
# baselines = {}
# baseline_names = {"ippo": ["ippo_sc2__seed*"]}
# for algo_name in ["ippo"]:
#     baselines[algo_name] = {}
#     for task_name in ["5m_vs_6m", "3s_vs_4z"]:
#         log_basenames = baseline_names[algo_name]
#         results = []
#         for logname in log_basenames:
#             print(os.path.join(base_paths[algo_name].format(task_name), logname))
#             results += glob.glob(os.path.join(base_paths[algo_name].format(task_name), logname))         
#         baselines[algo_name][task_name] = {f"{algo_name} baseline": results}
# baselines.keys()



# # dm2 core result experiment dict
# ippo_base_path = "/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/{}/"
# task_names = {# task_folder: basename_regex
#          "3s_vs_4z": "dm2-sa_sc2__seed=*", 
#          "5m_vs_6m": "dm2-sa_sc2__seed=*",
#          }
# # timesteps = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "10m"]

# experiments_core = {}
# for task_name, task_basename in task_names.items():
#     experiments_core[task_name] = {}
#     print(os.path.join(ippo_base_path.format(task_name), task_basename))
#     results = glob.glob(os.path.join(ippo_base_path.format(task_name), task_basename))
#     experiments_core[task_name]["5m"] = {f"dm2, ippo demo (5m)": results}
# experiments_core.keys()

# plot all results
# task_demo_dict = {
#                   "5m_vs_6m": ["5m"],
#                 #   "3s_vs_4z": ["5m"],
# }
# for task_name, demo_ts in task_demo_dict.items():
#     experiments = {}
#     # task_base = task_name.split("_")[0]
#     for ts in demo_ts:
#         experiments = {
#             **experiments,
#             **experiments_core[task_name][ts],
#         }

#     experiments = {
#                   **experiments,
#                    **baselines["ippo"][task_name], 
#     }


experiments_diff_coef = {

'PegMARL(diff, $\eta$=0.2)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=1285842_03-03-01-10-50',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=78590_03-03-01-10-50',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=112358_03-01-04-50-08',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=119527_03-03-01-11-02',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=122529_03-03-01-11-02',],

'PegMARL(diff, $\eta$=0.3)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen_sc2__seed=1285842_02-26-01-01-01',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen_sc2__seed=78590_02-26-01-01-01',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen_sc2__seed=112358_02-27-21-33-26',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen_sc2__seed=119527_02-26-01-01-01',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen_sc2__seed=122529_02-26-04-57-39'],


'PegMARL(diff, $\eta$=0.5)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.5_sc2__seed=1285842_03-04-21-53-51',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.5_sc2__seed=78590_03-04-22-40-18',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.5_sc2__seed=112358_03-04-21-20-22',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.5_sc2__seed=119527_03-04-23-11-23',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.5_sc2__seed=122529_03-04-21-54-56',]

}

experiments_coef = {
'PegMARL($\eta$=0.2)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=78590_03-03-01-12-44',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=112358_03-01-17-54-44',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=119527_03-03-01-36-24',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=122529_03-03-14-29-18',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=1285842_03-03-01-12-44',],

'PegMARL($\eta$=0.3)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen_sc2__seed=78590_02-27-22-17-35',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen_sc2__seed=112358_02-27-21-29-23',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen_sc2__seed=119527_02-27-22-31-02',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen_sc2__seed=122529_02-27-22-44-30',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen_sc2__seed=1285842_02-27-21-29-23',],

'PegMARL($\eta$=0.5)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.5-78590_sc2__seed=78590_03-04-17-28-08',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.5-112358_sc2__seed=112358_03-04-17-26-08',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.5-119527_sc2__seed=119527_03-04-17-29-12',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.5-122529_sc2__seed=122529_03-04-17-29-28',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.5-1285842_sc2__seed=1285842_03-04-17-28-08',],

}

experiments = {

'PegMARL': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=78590_03-03-01-12-44',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=112358_03-01-17-54-44',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=119527_03-03-01-36-24',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=122529_03-03-14-29-18',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-add-pen-rew0.2_sc2__seed=1285842_03-03-01-12-44',],

'PegMARL(diff)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=1285842_03-03-01-10-50',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=78590_03-03-01-10-50',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=112358_03-01-04-50-08',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=119527_03-03-01-11-02',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ours-nonco-add-pen-rew0.2_sc2__seed=122529_03-03-01-11-02',],

'DM2':  ['/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2_sc2__seed=78590_02-27-14-44-18',
 '/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2_sc2__seed=119527_02-27-15-25-39',
 '/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2_sc2__seed=122529_02-25-18-08-23',
 '/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2_sc2__seed=112358_02-27-09-37-31',
 '/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2_sc2__seed=1285842_02-27-10-19-56'],

'DM2(diff)':  [
'/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2-nonco_sc2__seed=122529_02-25-16-19-20',
'/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2-nonco_sc2__seed=119527_02-25-16-20-14',
'/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2-nonco_sc2__seed=78590_02-25-16-20-14',
'/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2-nonco_sc2__seed=1285842_02-25-16-20-22',
'/fs/nexus-projects/Guided_MARL/manav_dm2_results/tb_logs/sc2/5m_vs_6m/dm2-nonco_sc2__seed=112358_02-25-16-20-22'],

'IPPO': [
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_5v6_tb_logs/ippo_sc2_baseline_seed=1285842_01-20-14-29-52',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_5v6_tb_logs/ippo_sc2_baseline_seed=78590_01-20-14-30-52',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_5v6_tb_logs/ippo_sc2_baseline_seed=112358_01-20-14-29-52',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_5v6_tb_logs/ippo_sc2_baseline_seed=119527_01-20-14-30-52',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_5v6_tb_logs/ippo_sc2_baseline_seed=122529_01-20-14-30-52',],

# 'IPPO': [
# '/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ippo_sc2__seed=112358_02-21-01-55-31',
# '/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ippo_sc2__seed=119527_02-23-00-34-16',
# '/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ippo_sc2__seed=122529_02-23-00-34-23',
# '/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ippo_sc2__seed=1285842_02-21-01-55-31',
# '/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/5m_vs_6m/ippo_sc2__seed=78590_02-23-00-34-16'],

}

experiments_2 = {
'PegMARL': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-add-pen_sc2__seed=78590_03-01-21-18-45',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-add-pen_sc2__seed=112358_02-29-18-18-04',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-add-pen_sc2__seed=119527_03-01-23-15-15',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-add-pen_sc2__seed=122529_03-01-23-22-59',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-add-pen_sc2__seed=1285842_02-29-18-42-56',],

'PegMARL(diff)': [
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-nonco-add-pen_sc2__seed=78590_02-29-21-23-56',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-nonco-add-pen_sc2__seed=112358_02-28-20-47-01',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-nonco-add-pen_sc2__seed=119527_02-29-21-26-10',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-nonco-add-pen_sc2__seed=122529_02-29-23-05-11',
'/fs/nexus-projects/Guided_MARL/peihong_dm2_results/tb_logs/sc2/3s_vs_4z/ours-nonco-add-pen_sc2__seed=1285842_02-29-19-49-28',],

'DM2': [
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2_sc2__seed=78590_02-29-02-17-26',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2_sc2__seed=119527_02-29-07-05-15',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2_sc2__seed=122529_02-29-12-23-56',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2_sc2__seed=112358_02-29-12-50-29',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2-nonco_sc2__seed=122529_02-29-17-56-59'],

'DM2(diff)': [
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2-nonco_sc2__seed=1285842_02-28-16-47-15',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2-nonco_sc2__seed=112358_02-28-16-47-15',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2-nonco_sc2__seed=119527_02-29-16-06-01',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2-nonco_sc2__seed=78590_02-29-03-33-27',
'/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/dm2-nonco_sc2__seed=122529_02-29-17-56-59'],

'IPPO': [
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_3sv4z_tb_logs/ippo_sc2_baseline_seed=1285842_01-23-20-41-57',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_3sv4z_tb_logs/ippo_sc2_baseline_seed=112358_01-23-20-41-57',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_3sv4z_tb_logs/ippo_sc2_baseline_seed=119527_01-23-20-41-57',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_3sv4z_tb_logs/ippo_sc2_baseline_seed=122529_01-23-20-41-57',
'/fs/nexus-projects/Guided_MARL/dm2_demonstrations/ippo_3sv4z_tb_logs/ippo_sc2_baseline_seed=78590_01-23-20-41-57',],

# 'IPPO': [
# '/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/ippo_sc2__seed=112358_03-02-16-52-09',
# '/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/ippo_sc2__seed=122529_03-02-16-52-10',
# '/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/ippo_sc2__seed=1285842_03-01-18-39-32',
# '/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/ippo_sc2__seed=119527_03-01-18-15-18',
# '/fs/nexus-projects/Guided_MARL/dm2_results/tb_logs/sc2/3s_vs_4z/ippo_sc2__seed=78590_03-01-18-15-18']
}

# task_name = "3s_vs_4z"
# exp_dfs = gen_log_data(experiments_2)

# task_name = "5m_vs_6m"
# exp_dfs = gen_log_data(experiments)

task_name = "5m_vs_6m"
exp_dfs = gen_log_data(experiments_diff_coef)

plot_experiments(exp_dfs, 
                plot_title=f"{task_name}",
                savename=f"{task_name}_coef_diff_exp", #"qmix policies"
                demo_qual=demo_win_rate[task_name],
                custom_color_order=[2,1,3,0,4, 5],
                legend_cols=2,
                legend_loc=(1.15, -0.25),
                yaxis_lims=(-0.05, 0.85),
                save=True,
                savedir="")
