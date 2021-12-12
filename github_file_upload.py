import numpy as np
import simpy
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import math

 #------to add-----:

# Make the alternative scenario with New Loss Rate and eventual change in Demand. 
# I assume that after the optimization and so the optimal crop mix, the producer rents the land and hires workers
# so only for subsequent optimization calls there will be the land and workers as new constraints (to accept new demand if in stage_1 == seeding)

 # -----------------------------------------------------------------------------------------------------------------------------------------#

# EDSS # 
# LC stands for Life Cycle
# The time units are expressed in minutes and the quantities in lettuce or cauliflower heads
# Lettuce in an ideal scenario takes 45 days to be ready in open field and cauliflower 70 days
# Both veggies LC stages are divided in activity and queue, respectively the activity time is always 25% of total time except for last stage

edss_lettuce = { 'upper_loss_rate' : [0.2,0.2,0.3,0.4], 'lower_loss_rate' : [0.1,0.1,0.05,0.1], 'upper_new_loss_rate' : [0.2,0.2,0.3,0.4],'lower_new_loss_rate' : [0.1,0.1,0.05,0.1],
'max_attainable_per_hectar': 1350, 'demand': np.random.randint(8000, 12000),"activity_step":1000,"Yield/10k_hectars": 2000,"Time_per_lc_stage":[4320,10080,14400,28800], 
'stages_interval':[70,50,40,50],'farming_time':[2,2,1,24]}

edss_cauliflower = { 'upper_loss_rate' : [0.3,0.4,0.4,0.55], 'lower_loss_rate' : [0.2,0.2,0.3,0.4],'upper_new_loss_rate' : [0.3,0.4,0.4,0.55], 'lower_new_loss_rate' : [0.2,0.2,0.3,0.4],
'max_attainable_per_hectar': 1500, 'demand': np.random.randint(8000, 12000),"activity_step":1000,"Yield/10k_hectars": 3000,"Time_per_lc_stage":[6480,15120,21600,43200], 
'stages_interval':[60,100,70,50],'farming_time':[3,1,1.5,18]}

# Variables
activity_time = 0.25
queue_time = 1 - activity_time
total_workers_available = 10
total_demand = np.random.randint(edss_lettuce['demand'] + edss_cauliflower['demand'], 30000)

# Weights in crop mix and scheduling decision making
min_visible_waste_weight = 0.7
min_total_input_weight = 0.3

# Lower bound mean
lettuce_mean_loss_rate_lower_bound = np.mean(edss_lettuce ['lower_loss_rate']).mean()
cauliflower_mean_loss_rate_lower_bound = np.mean(edss_cauliflower ['lower_loss_rate']) 

# Coefficients of model/objective 2
lettuce_mean_loss_rate_upper_bound = np.mean(edss_lettuce ["upper_loss_rate"]) 
cauliflower_mean_loss_rate_upper_bound = np.mean(edss_cauliflower["upper_loss_rate"])

# Coefficients of model/objective 1
lettuce_variability = lettuce_mean_loss_rate_upper_bound - lettuce_mean_loss_rate_lower_bound
cauliflower_variability = cauliflower_mean_loss_rate_upper_bound - cauliflower_mean_loss_rate_lower_bound


    # ---- CROP MIX AND SCHEDULING OPTIMIZATION -----

#Target 1
# MINIMIZE TOTAL WASTE (IMPACT PER INPUT BY MINMIZING PRODUCTION OF GOODS WITH A WIDER LOSS RATE RANGE (VARIABILITY))
model_1 = Model(name = "lp_Expected_Waste_Target")  

lettuce = model_1.addVar ( name = "lettuce", vtype = GRB.CONTINUOUS, lb = 0)
cauliflower = model_1.addVar ( name = "cauliflower", vtype = GRB.CONTINUOUS, lb = 0)

obj_1 = lettuce_variability*lettuce + cauliflower_variability*cauliflower

model_1.setObjective ( obj_1, GRB.MINIMIZE)  #define obj function and "the sense"

#Subject to:

c1 = model_1.addConstr ( lettuce + cauliflower >= total_demand, name = "c1" )
c2 = model_1.addConstr ( lettuce  >= edss_lettuce['demand'],  name = "c2" )    
c3 = model_1.addConstr (cauliflower >= edss_cauliflower['demand'], name = "c3" )
#Non negativity constraint is embed in the lower-bound when the variable has been defined above

model_1.optimize()
# model_1.write (filename)  write model to a file

print ( 'Obj_1 Function Value - Expected_Waste_Target : %f' % model_1.objVal)

# Get values of the decision variables
lettuce_results = []
cauliflower_results = []
objs = []
lettuce_results.append(model_1.X[0])
cauliflower_results.append(model_1.X[1])
objs.append(model_1.objVal)
# for e in model_1.getVars () :
#     print ( "%s: %g" % ( e.varName, e.x))
    
# ----------------------------------------------------#

#Target 2
# MINIMIZE TOTAL INPUT REQUIRED BY MINIMIZING THE PRODUCTION OF THE WORST PERFORMING PRODUCTS 
# (The one with the higher production loss rate upper bound which tranlates in greater inputs requirements)

model_2 = Model (name = "lp_Tot_Exp_Loss_Target") 

lettuce = model_2.addVar ( name = "lettuce", vtype = GRB.CONTINUOUS, lb = 0)
cauliflower = model_2.addVar ( name = "cauliflower", vtype = GRB.CONTINUOUS, lb = 0)

obj_2 = lettuce_mean_loss_rate_upper_bound*lettuce + cauliflower_mean_loss_rate_upper_bound*cauliflower
model_2.setObjective ( obj_2, GRB.MINIMIZE)  #define obj function and "the sense"

#Subject to:

c1 = model_2.addConstr ( lettuce + cauliflower >= total_demand, name = "c1" )
c2 = model_2.addConstr ( lettuce  >= edss_lettuce['demand'],  name = "c2" )    
c3 = model_2.addConstr (cauliflower >= edss_cauliflower['demand'], name = "c3" )
#Non negativity constraint is embed in the lower-bound when the variable has been defined above

model_2.optimize()
# model_1.write (filename)  write model to a file

print ( 'Obj_2 Function Value - Tot_Exp_Loss_Target: %f' % model_2.objVal)

# Get values of the decision variables

lettuce_results.append(model_2.X[0])
cauliflower_results.append(model_2.X[1])
objs.append(model_2.objVal)
for e in model_2.getVars () :
    print ( "%s: %g" % ( e.varName, e.x))  

# --------------------------------------------------------- #

goal_programming_model = Model(name = "Goal Programming - Minimize sum of max percentage deviation ") 

lettuce = goal_programming_model.addVar ( name = "lettuce", vtype = GRB.CONTINUOUS, lb = 0)
cauliflower = goal_programming_model.addVar ( name = "cauliflower", vtype = GRB.CONTINUOUS, lb = 0)
Q = goal_programming_model.addVar ( name = " Q", vtype = GRB.CONTINUOUS, lb = 0)


multi_obj_deviations = Q #minMax variable  ( minimize sum of max percentage deviation )
goal_programming_model.setObjective ( multi_obj_deviations, GRB.MINIMIZE)  #define obj function and "the sense"

#Subject to:

#Previous targets now constraints:
c4 = goal_programming_model.addConstr (min_visible_waste_weight *((lettuce_variability* lettuce + cauliflower_variability*cauliflower)-model_1.objVal)/model_1.objVal  <= Q , name = "obj_1")
c5 = goal_programming_model.addConstr( min_total_input_weight *(( (lettuce_mean_loss_rate_upper_bound* lettuce) + (cauliflower_mean_loss_rate_upper_bound* cauliflower))-model_2.objVal)/model_2.objVal <= Q, name = "obj_2" )

#Previous constraints:    

c1 = goal_programming_model.addConstr ( lettuce + cauliflower >= total_demand, name = "c1" )
c2 = goal_programming_model.addConstr ( lettuce  >= edss_lettuce['demand'],  name = "c2" )    
c3 = goal_programming_model.addConstr (cauliflower >= edss_cauliflower['demand'], name = "c3" )
#Non negativity constraint is embed in the lower-bound when the variable has been defined above

goal_programming_model.optimize()
# model_1.write (filename)  write model to a file

lettuce_results.append(goal_programming_model.X[0])
cauliflower_results.append(goal_programming_model.X[1])
objs.append(goal_programming_model.objVal)


# for e in goal_programming_model.getVars () :
#     print ( "%s: %g" % ( e.varName, e.x))
   
print ("Lettuce vector", lettuce_results)   
print ("Cauliflower vector", cauliflower_results) 
print ("Obj vector",objs)

#------------------------------ PLOT ---------------------------#

minMax_result_target_1 = lettuce_variability*lettuce_results [2] + cauliflower_variability*cauliflower_results [2]
minMax_result_target_2 = lettuce_mean_loss_rate_upper_bound*lettuce_results [2] + cauliflower_mean_loss_rate_upper_bound*cauliflower_results [2]

plt.figure(figsize=(10,8))

d = np.linspace(0,30000,50)
x,y = np.meshgrid(d,d)

print(x)
print(y)

plt.imshow((x>=edss_lettuce['demand']) & (y>=edss_cauliflower['demand']) & (x+y>=total_demand), 
          extent=(x.min()+1,x.max()+1,y.min()+1,y.max()+1),origin="lower", cmap="Greys", alpha = 0.3)  # Feasibile area

# plt.figure()
plt.xlim(5000,total_demand)
plt.ylim(5000,total_demand)
plt.hlines(edss_cauliflower['demand'], 0,total_demand, linestyles='--', alpha=0.6)
plt.vlines(edss_lettuce['demand'], 0,total_demand, linestyles='--', alpha=0.6)
plt.plot([0,total_demand],[total_demand,0], linestyle='--', label='constraints', alpha=0.6)

# Corner solutions plot
# plt.plot([model_1.objVal/lettuce_variability,0],[0,model_1.objVal/cauliflower_variability], 'brown', label='model_1 (min exp waste', linewidth=3) #Solution of model 1
# plt.plot([model_2.objVal/lettuce_mean_loss_rate_upper_bound,0], [0,model_2.objVal/cauliflower_mean_loss_rate_upper_bound], 'hotpink', label='model_2 min tot input', lw=3)  #Solution of model 2

# minMax plot
plt.plot([minMax_result_target_1/lettuce_variability,0],[0,minMax_result_target_1/cauliflower_variability], "blue", label='model 1 minMax result', linewidth=3) #Solution of model 1
plt.plot([minMax_result_target_2/lettuce_mean_loss_rate_upper_bound,0], [0,minMax_result_target_2/cauliflower_mean_loss_rate_upper_bound], 'red', label='model 2 minMax result', lw=3)  #Solution of model 2


plt.annotate('Feasable area', xy=(10000,10000))
plt.legend()
plt.xlabel('lettuce',size = 10)
# plt.show()

# OPTIMIZATION RESULTS:
optimization_results = {'lettuce_crop_optimal':math.ceil(lettuce_results [2]), 'cauliflower_crop_optimal':math.ceil(cauliflower_results [2])}
optimization_results['lettuce_workers'] = round(optimization_results['lettuce_crop_optimal']*total_workers_available/total_demand)
optimization_results['cauliflower_workers'] = total_workers_available - optimization_results['lettuce_workers']
print('OPTIMIZATION RESULT \n', optimization_results)
plt.show()     

# CALCULATE OPTIMAL INPUT GIVEN THE ABOVE CROP MIX DECISION AND THE UPPER BOUND OF THE PRODUCTION YIELD LOSS RATE CALLED WORST CASE SCENARIO (WCs)
# The farmer is risk adverse and always plan for the WCS
# Class "Veggie" stores the Veggies attributes and production data 
# Class "S3" is the digital twin of the real environment and the continuous growth process and stochastic set of activity is tracked in 4 discrete states
# alternating a time window of activity where workers are required and a time window of queue where no work is needed on the plant and workers are free.
# Workers are the resources, divided in two pools and are dedicated. 
# S3 explores alterantive scenarios to optimize human resources usage during the productions and eventually making them floating.
# Real_loss_rate is a variable in the S3 method "Farming" that simulate the real world stochasticity
                             ----------------------------------------------------------------------------
class veggie():

    def __init__(self, est_yield, stages_interval, farming_time, name, actvivity_step, edss):
        self.name = name
        self.est_yield = est_yield
        self.stages_interval = stages_interval
        self.farming_time = farming_time
        self.activity_step = actvivity_step
        self.edss = edss
        self.stages_dic = {"farmed":0, "fertilized" :0, 'blossomed':0, 'harvested':0}
        self.plot_dic = {"farmed": {'time':[], 'quantity':[]}, 'fertilized': {'time':[], 'quantity':[]},
        'blossomed': {'time':[], 'quantity':[]}, 'harvested': {'time':[], 'quantity':[]}}

    # def loss_rate(self, veg, path=r'C:\Users\Jacop\Downloads\excel.xls'):
    #     original_loss_rate = pd.read_excel(path, sheet_name=f'{veg}_1',usecols=['Up'], squeeze=True)[:4]
    #     new_loss_rate =  pd.read_excel(path, sheet_name=f'{veg}_2',usecols=['Up'], squeeze=True)[:4]
    #     if original_loss_rate.any() == new_loss_rate.any() :
    #         #and stage == 1 (add stage as argument) and if lettuce_demand,cauliflower_demand and total_deamand
    #         # if veg == "Lettuce":
    #         #     optimal_input = calc_optimal_input (lettuce_crop_optimal,edss_lettuce['demand'])
    #         return [original_loss_rate, 0]
    #     else:
    #         # if veg == "Lettuce":
    #         #     optimal_input = calc_optimal_input (lettuce_crop_optimal,edss_lettuce['demand'])
    #             # recalculate what should be the input now
    #             # make the difference on what should be planted now, run optimization for new optimal crop mix and proceed
    #             # recalcola optimal_input/est_yield -> new_opt_input - self.stages_dic['farmed']
    #        # if veg == "cauliflower"
    #          # do the same
            # return [new_loss_rate, 1]

    def tot_farm_time(self, n_workers_initial, n_workers_new, stage):

        # total time obtained doing est_yield divided by activity step (1000) * farming_time in each step (i) diveded initial workers (7 or 3)
        total_farming_time_initial = [(i*(self.est_yield/self.activity_step))/n_workers_initial for i in self.farming_time]

        # total time obtained doing est_yield divided by activity step (1000) * farming_time in each step (i) diveded NEW workers (10)
        total_farming_time_new = [(i*(self.est_yield/self.activity_step))/n_workers_new for i in self.farming_time]

        # GEtting percentage change for specific state
        perc_change = abs([1-((new/old)*100) for new, old in zip(total_farming_time_new, total_farming_time_initial)][stage])/100

                # Return a list of new farming time if there were 10 workers in each stage
        return [np.array(self.farming_time)*(1-perc_change), perc_change]


class s3():
    
    def __init__(self):

        self.optimal_input_lettuce = self.calc_optimal_input(optimization_results['lettuce_crop_optimal'],edss_lettuce["upper_loss_rate"])
        self.optimal_input_cauliflower =  self.calc_optimal_input(optimization_results['cauliflower_crop_optimal'],edss_cauliflower['upper_loss_rate'])

        self.env = simpy.rt.RealtimeEnvironment(factor=0.01, strict=False)
        self.lettuce = veggie(est_yield = self.optimal_input_lettuce, stages_interval=edss_lettuce['stages_interval'], farming_time=edss_lettuce['farming_time'],name='Lettuce', actvivity_step=edss_lettuce['activity_step'], edss=edss_lettuce)
        self.cauli = veggie(est_yield = self.optimal_input_cauliflower,stages_interval=edss_cauliflower['stages_interval'], farming_time=edss_cauliflower['farming_time'], name='Cauliflower', actvivity_step=edss_cauliflower['activity_step'], edss=edss_cauliflower)
        self.farmer_lettuce = simpy.Resource(self.env, capacity=1)
        self.farmer_cauli = simpy.Resource(self.env, capacity=1)
        # land = simpy.Container(env, capacity= enough to satify WCS optimal input, init= stesso di capacity)

        # Atlernative scenario resources 
        self.env_fast = simpy.rt.RealtimeEnvironment(factor=0.001, strict=False, initial_time=0)
        self.lettuce_fast = veggie(est_yield = self.optimal_input_lettuce,  stages_interval=edss_lettuce['stages_interval'], farming_time=edss_lettuce['farming_time'], name='Lettuce', actvivity_step=edss_lettuce['activity_step'], edss=edss_lettuce)
        self.cauli_fast = veggie(est_yield = self.optimal_input_cauliflower, stages_interval=edss_cauliflower['stages_interval'], farming_time=edss_cauliflower['farming_time'], name='Cauliflower', actvivity_step=edss_cauliflower['activity_step'], edss=edss_cauliflower)
        self.farmer_lettuce_fast = simpy.Resource(self.env_fast, capacity=1)
        self.farmer_cauli_fast = simpy.Resource(self.env_fast, capacity=1)
        
        self.stage = -1
    

    def farming(self, env, veg, farmer, farmtime, stage):
        # # STAGE 0
        # stage = -1
        stages = ['farmed', 'fertilized','blossomed','harvested']

        waste_per_stage = {k:0 for k in stages}

        while stage<3:
            if env == self.env_fast:
                print('time', env.now)

            stage+=1
            print(f'STAGE {stage} of {veg.name} STARTS AT {env.now}')
            while veg.stages_dic[stages[stage]] < veg.est_yield:
                # check if there is new demand or loss rate different or if demand is different
                # if stage == 1:
                #     veg.loss_rate()
                # else:
                    # veg.loss_rate()
                if veg.est_yield-veg.stages_dic[stages[stage]] < veg.activity_step:
                    with farmer.request() as req:
                        yield req
                        veg.stages_dic[stages[stage]]+= (veg.est_yield-veg.stages_dic[stages[stage]])+1
                        veg.plot_dic[stages[stage]]['quantity'].append(veg.stages_dic[stages[stage]])
                        veg.plot_dic[stages[stage]]['time'].append(env.now)
                        yield env.timeout(farmtime[stage])
                         
                else:
                    with farmer.request() as req:
                        yield req
                        veg.stages_dic[stages[stage]]+=veg.activity_step
                        veg.plot_dic[stages[stage]]['quantity'].append(veg.stages_dic[stages[stage]])
                        veg.plot_dic[stages[stage]]['time'].append(env.now)
                        yield env.timeout(farmtime[stage])
            print(f'time {round(env.now, 2)} to {stages[stage]} {str(veg.name).upper()}: ACTIVITY FINISHED!\n')

            # Percentage increase/decrease farming time
            perc_lettuce = veg.tot_farm_time(n_workers_initial=3, n_workers_new=10, stage=stage)[1]
            perc_cauli = veg.tot_farm_time(n_workers_initial=7, n_workers_new=10, stage=stage)[1]

            time_after_activity = env.now
            


            # MOVE POOL OF WORKERS THAT ARE FREE EARLIER 
            if self.lettuce.stages_dic[stages[stage]] >= self.lettuce.est_yield and self.cauli.stages_dic[stages[stage]]< self.cauli.est_yield and env != self.env_fast:
                self.alternative_scenarios(stage=stage-1, veg='cauli')
                print('ALTERNATIVE SCENARIO END')
                print('- - -'*30)
                decision = input(f'Idle resources detected. Alternative scenario analysis show time saving of {perc_cauli*100}% if resourses are moved. Proceed? Yes/No ')
                if decision.lower() in ['yes','y']:
                    farmtime[stage] *= (1-perc_cauli)
                    # call tot farm time function so that it will return the % change precicely - this is lettuce workers moving to cauli, so 7->10
                    print(f'\nYou have saved {perc_cauli*100}% on activity time for cauli')

                    # In case lettuce queue finishes while pool of workers is still helping out for cauli, they are called back to work on lettuce
                    if env.now >= time_after_activity + self.cauli.stages_interval[stage]:
                        print('Pool of resources back to original vegetable')
                        farmtime[stage] /= (1-perc_cauli)
            
            if self.cauli.stages_dic[stages[stage]] >= self.cauli.est_yield and  self.lettuce.stages_dic[stages[stage]] < self.lettuce.est_yield and env != self.env_fast:
                self.alternative_scenarios(stage=stage-1, veg='lettuce')
                print('ALTERNATIVE SCENARIO END')
                print('- - -'*30)
                decision = input(f'Idle resources detected. Alternative scenario shows that moving resources you can save {perc_lettuce*100}% time. Do you want to move? ')
                if decision.lower() in ['yes','y']:
                    farmtime[stage] *= (1-perc_lettuce)
                    # call tot farm time function so that it will return the % change precicely - this is cauli workers moving to lettuce, so 3->10
                    print(f'\nYou have saved {perc_lettuce}% on activity time for lettuce')

                    # In case cauli queue finishes while pool of workers is still helping out for lettuces, they are called back to work on cauli
                    if env.now >= time_after_activity + self.lettuce.stages_interval[stage]:
                        print('Pool of resources back to original vegetable')
                        farmtime[stage] /= (1-perc_lettuce)

            # # QUEUE - difference between the queue estimated(scheduled time) and the time that has already elapsed while the pool of a veggie was sent to helo the other veggie production
            yield env.timeout(veg.stages_interval[stage] - (env.now - time_after_activity))
            print(f'Queue done! {veg.name, env.now} stage {stage} completed')
            print ("  ")
            real_loss_rate = np.random.uniform(veg.edss['lower_loss_rate'][stage], veg.edss['upper_loss_rate'][stage])
            waste_per_stage[stages[stage]] = abs((veg.est_yield * (1-veg.edss['upper_loss_rate'][stage])) - veg.est_yield * (1- real_loss_rate))

            if stage>=0 and stage <= 2:
                # real_loss_rate = np.random.uniform(veg.edss['lower_loss_rate'][stage], veg.edss['upper_loss_rate'][stage])
                # waste_per_stage[stages[stage]] = abs((veg.est_yield * (1-veg.edss['upper_loss_rate'][stage])) - veg.est_yield * (1- real_loss_rate))
                # Update EST YIELD
                veg.est_yield = veg.est_yield*((1-veg.edss['upper_loss_rate'][stage]))
                print(f'\n EST YIELD in stage {stage, veg.name} {veg.est_yield} \n')
                print(veg.name, ':' ,veg.stages_dic, sep=' ')
            
            elif stage == 3:

                veg.stages_dic["estimated_harvested_product"] = veg.est_yield*((1-veg.edss['upper_loss_rate'][stage]))
                veg.stages_dic["actual_harvested_product"] = veg.est_yield - (veg.est_yield * real_loss_rate)
                veg.stages_dic["VISIBILE_OVERPRODUCTION (WASTE) "] = abs((veg.est_yield*((1-veg.edss['upper_loss_rate'][stage]))) - (veg.est_yield - (veg.est_yield * real_loss_rate)))
                print (veg.stages_dic)
                print ("----------------")
                print('EXCESS IN INITIAL INPUT (SEEDS)', abs(waste_per_stage ["farmed"]))
                print ("----------------")
                print ("----------------")
           

        # #  PLOTS
        #     if stage == 3 and env != self.env_fast:
        #         fig, ax = plt.subplots(8, figsize=(3,8))
        #         p = -1
        #         for s in veg.plot_dic.keys():
        #             for t,q in veg.plot_dic[s].items():
        #                 p+=1
        #                 # print(f"stage: {s} time {veg.plot_dic[s]['time']} quantity {veg.plot_dic[s]['quantity']}")
        #                 ax[p].plot(veg.plot_dic[s]['time'], veg.plot_dic[s]['quantity'])
        #                 ax[p].set_title(f'{veg.name} in stage {s}', fontsize=8)


    # def calc_optimal_input (lettuce_crop_optimal,cauliflower_crop_optimal,edss_lettuce["upper_loss_rate"],edss_cauliflower["upper_loss_rate"]):
    def calc_optimal_input (self,crop_optimal, loss_rates):
        gamma = 1 - loss_rates[0]
        for i in loss_rates[1:] :
            gamma = gamma * (1 - i)  
        optimal_input = crop_optimal / gamma
        return optimal_input 

    def alternative_scenarios(self, stage, veg):
        print('- - -'*30)
        print("ALTERNATIVE SCENARIO START")
        new_farmtime_cauli = self.cauli.tot_farm_time(n_workers_initial=7, n_workers_new=10, stage=stage)[0]
        new_farmtime_lettuce = self.lettuce.tot_farm_time(n_workers_initial=3, n_workers_new=10, stage=stage)[0]

        if veg.lower() in ['cauli', 'cauliflower']:
            self.env_fast.process(self.farming(self.env_fast, self.cauli_fast, self.farmer_cauli_fast, new_farmtime_cauli, stage))
        else:
            self.env_fast.process(self.farming(self.env_fast, self.lettuce_fast, self.farmer_lettuce_fast, new_farmtime_lettuce, stage))
        self.env_fast.run()
    
    def generate(self):
        self.env.process(self.farming(self.env, self.cauli, self.farmer_cauli, self.cauli.farming_time, self.stage))
        self.env.process(self.farming(self.env, self.lettuce, self.farmer_lettuce, self.lettuce.farming_time, self.stage))
        self.env.run()        



farm = s3()
farm.generate()
# plt.show()
