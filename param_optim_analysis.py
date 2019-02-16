import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

files= os.listdir("test_out_2")
p_files=[]
best_scores=[]
BUFFER_SIZE=[]
BATCH_SIZE=[]
GAMMA=[]
TAU=[]
LR_ACTOR=[]
LR_CRITIC=[]
WEIGHT_DECAY=[]
LEARN_EVERY=[]
ACTOR_FC1=[]
ACTOR_FC2=[]
CRITIC_FC1=[]
CRITIC_FC2=[]

for f in files:
    if f[-2:]==".p":
        p_files.append(f)
        score_list, mean_score_list, pars = pickle.load(open("test_out_2/"+f, "rb"))
        solved_epoch=10000000
        solved=False
        best_score = 0
        for epoch in range(len(mean_score_list)):
            if mean_score_list[epoch] >13.0:
                if not solved:
                    solved_epoch=100*epoch
                    solved=True
            best_score=max(best_score, mean_score_list[epoch])

        best_scores.append(best_score)
        BUFFER_SIZE.append(pars['BUFFER_SIZE'])
        BATCH_SIZE.append(pars['BATCH_SIZE'])
        GAMMA.append(pars['GAMMA'])
        TAU.append(pars['TAU'])
        LR_ACTOR.append(pars['LR_ACTOR'])
        LR_CRITIC.append(pars['LR_CRITIC'])
        WEIGHT_DECAY.append(pars['WEIGHT_DECAY'])
        LEARN_EVERY.append(pars['LEARN_EVERY'])
        ACTOR_FC1.append(pars['ACTOR_FC1'])
        ACTOR_FC2.append(pars['ACTOR_FC2'])
        CRITIC_FC1.append(pars['CRITIC_FC1'])
        CRITIC_FC2.append(pars['CRITIC_FC2'])


res_dict={"file_name": p_files, "best_score": best_scores, "BUFFER_SIZE":BUFFER_SIZE, "BATCH_SIZE":BATCH_SIZE,
          "GAMMA":GAMMA, "TAU":TAU, "LR_ACTOR":LR_ACTOR,
          "LR_CRITIC":LR_CRITIC, "WEIGHT_DECAY": WEIGHT_DECAY, "LEARN_EVERY": LEARN_EVERY, "ACTOR_FC1":ACTOR_FC1,
          "ACTOR_FC1":ACTOR_FC1,"CRITIC_FC1":CRITIC_FC1,"CRITIC_FC2":CRITIC_FC2}

res_df=pd.DataFrame(res_dict)

res_df=res_df.sort_values("best_score", ascending=False)
res_df.to_csv("optim_results.csv")


f= res_df.file_name[4]
score_list, mean_score_list, pars = pickle.load(open("test_out_2/" + f, "rb"))
plt.figure()
plt.plot(score_list)
x = range(len(mean_score_list))
plt.plot(25+np.array(x)*50.0, mean_score_list)
plt.ylabel('Score')
plt.xlabel('Episode #')

# colors= ["ro", "go", "rs", "gs" ]
#
#
# plt.figure() #mem_size: 5000+
# plt.plot(res_df["mem_size"].astype(float), res_df["best_score"], "ro")
# plt.figure() #learn every 4 or 16?
# plt.plot(res_df["learn_every"].astype(float), res_df["best_score"], "ro")
# plt.figure()
# plt.plot(res_df["double_qnet"].astype(float), res_df["best_score"], "ro")
# plt.figure()
# plt.plot(res_df["delayer"].astype(float), res_df["best_score"], "ro")
# plt.figure()
# plt.plot(res_df["eps_decay"].astype(float), res_df["best_score"], "ro")
#
# best_res_df=res_df[res_df["mem_size"]>=5000]
# best_res_df=best_res_df[res_df["eps_decay"]==0.99]
# best_res_df=best_res_df[res_df["learn_every"]==4]
# best_res_df=best_res_df[res_df["update_every"]<10]
# best_res_df=best_res_df[res_df["delayer"]==True]
# best_res_df=best_res_df[res_df["double_qnet"]==True]
#
# plt.figure() # update every:2
# plt.plot(best_res_df["update_every"].astype(float), best_res_df["best_score"], "ro")
#
# plt.figure() #learning rate 0.005
# plt.plot(best_res_df["learning_rate"].astype(float), best_res_df["best_score"], "ro")