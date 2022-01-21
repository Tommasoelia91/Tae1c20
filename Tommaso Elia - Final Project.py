import numpy as np
import simpy
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import *
import math

np.random.seed(42) #for validation

# Make the alternative scenario with New Loss Rate and eventual change in Demand. 
# I assume that after the optimization and so the optimal crop mix, the producer rents the land and hires workers
# so only for subsequent optimization already_fulls there will be the land and workers as new constraints (to accept new demand if in stage_1 == seeding)

 # -----------------------------------------------------------------------------------------------------------------------------------------#

# EDSS # 
# LC stands for Life Cycle
# The time units are expressed in minutes and the quantities in veg_1 or veg_2 heads
# veg_1 in an ideal scenario takes 45 days to be ready in open field and veg_2 70 days
# Both veggies LC stages are divided in activity and queue, respectively the activity time is always 25% of total time except for last stage


print('\n\n')

n_plot = 1
def optimization(veg_1_edss, veg_2_edss, total_workers_available, total_demand, land_veg_1 = None, land_veg_2=None, sowed=None, original_1=0, original_2=0, too_much=False):
    global n_plot
    # Weights in crop mix and scheduling decision making
    min_visible_waste_weight = 0.7
    min_total_input_weight = 0.3

    # Lower bound mean
    veg_1_mean_loss_rate_lower_bound = np.mean(veg_1_edss['lower_loss_rate'])
    veg_2_mean_loss_rate_lower_bound = np.mean(veg_2_edss['lower_loss_rate']) 

    # Coefficients of model/objective 2
    veg_1_mean_loss_rate_upper_bound = np.mean(veg_1_edss["upper_loss_rate"]) 
    veg_2_mean_loss_rate_upper_bound = np.mean(veg_2_edss["upper_loss_rate"])

    # Coefficients of model/objective 1
    veg_1_variability = veg_1_mean_loss_rate_upper_bound - veg_1_mean_loss_rate_lower_bound
    veg_2_variability = veg_2_mean_loss_rate_upper_bound - veg_2_mean_loss_rate_lower_bound

    print(f'OPTIMIZATION STARTING NOW, DEMAND {veg_1_edss["name"]} :', veg_1_edss['demand'], 'DEMAND {veg_2_edss["name"]}:', veg_2_edss['demand'], 'TOTAL DEMAND',total_demand)

        # ---- CROP MIX AND SCHEDULING OPTIMIZATION -----

    #Target 1
    # MINIMIZE TOTAL WASTE (IMPACT PER INPUT BY MINMIZING PRODUCTION OF GOODS WITH A WIDER LOSS RATE RANGE (VARIABILITY))
    model_1 = Model(name = "lp_Expected_Waste_Target")  
    model_1.Params.LogToConsole = 0

    veg_1 = model_1.addVar ( name = "veg_1", vtype = GRB.CONTINUOUS, lb = 0)
    veg_2 = model_1.addVar ( name = "veg_2", vtype = GRB.CONTINUOUS, lb = 0)

    obj_1 = veg_1_variability*veg_1 + veg_2_variability*veg_2

    model_1.setObjective ( obj_1, GRB.MINIMIZE)  #define obj function and "the sense"

    #Subject to:
    # First set of constraints for the first optimization when land has not been booked yet
    if too_much == False and land_veg_1==None and land_veg_2==None:
        c1 = model_1.addConstr ( veg_1 + veg_2 >= total_demand, name = "c1" )
        c2 = model_1.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = model_1.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )

    # This is set of constraints for when the land has already been booked and sowed paartilly and there is space to satisfy further demand
    elif too_much==False and land_veg_1 != None and land_veg_2!=None:
        # This constraint is dynamic - the sum of the optimal inputs (of seeds) for the crop mix found needs to require less land than the land available 
        #  because land is function of seeds - func(seeds) = land

        c6 = model_1.addConstr((s3.calc_optimal_input(crop_optimal=veg_2, loss_rates=veg_2_edss['upper_loss_rate']) / veg_2_edss['plants_hact'] + 
        s3.calc_optimal_input(crop_optimal=veg_1, loss_rates=veg_1_edss['upper_loss_rate']) / veg_1_edss['plants_hact']) 
        <= ((land_veg_1*veg_1_edss['squeezing_perc'])+(land_veg_2*veg_2_edss['squeezing_perc'])), name='c6')
        c1 = model_1.addConstr ( veg_1 + veg_2 >= total_demand, name = "c1" )
        c2 = model_1.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = model_1.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )
        # c4 = model_1.addConstr((veg_1+veg_2)<=(original_1*1.3+original_2*1.3))

        #Non negativity constraint is embed in the lower-bound when the variable has been defined above

    # set of constrains for when demand is too big to be satisfied and all land gets used and squeezed - too_much==True - and so it it maximised automatically
    elif too_much==True:
        c1 = model_1.addConstr ( veg_1  >=  veg_1_edss['demand'],  name = "c1" )    
        c2 = model_1.addConstr (veg_2 >= veg_2_edss['demand'], name = "c2" )
        # c4 = model_1.addConstr((veg_1+veg_2)==(original_1*1.3+original_2*1.3))
        c3 = model_1.addConstr((s3.calc_optimal_input(crop_optimal=veg_2, loss_rates=veg_2_edss['upper_loss_rate']) / veg_2_edss['plants_hact'] + 
        s3.calc_optimal_input(crop_optimal=veg_1, loss_rates=veg_1_edss['upper_loss_rate']) / veg_1_edss['plants_hact'])
        == ((land_veg_1*veg_1_edss['squeezing_perc'])+(land_veg_2*veg_2_edss['squeezing_perc'])), name='c4')
        



    
    model_1.optimize()

    # Get values of the decision variables
    veg_1_results = []
    veg_2_results = []
    objs = []
    veg_1_results.append(model_1.X[0])
    veg_2_results.append(model_1.X[1])
    objs.append(model_1.objVal)


    # In case user wants to see value found by each optimization function, only uncomment next two lines to print
    # for e in model_1.getVars () :
        # print ( "%s: %g" % ( e.varName, e.x))
        
    # ----------------------------------------------------#

    #Target 2
    # MINIMIZE TOTAL INPUT REQUIRED BY MINIMIZING THE PRODUCTION OF THE WORST PERFORMING PRODUCTS 
    # (The one with the higher production loss rate upper bound which tranlates in greater inputs requirements)

    model_2 = Model (name = "lp_Tot_Exp_Loss_Target") 
    model_2.Params.LogToConsole = 0

    veg_1 = model_2.addVar ( name = "veg_1", vtype = GRB.CONTINUOUS, lb = 0)
    veg_2 = model_2.addVar ( name = "veg_2", vtype = GRB.CONTINUOUS, lb = 0)

    obj_2 = veg_1_mean_loss_rate_upper_bound*veg_1 + veg_2_mean_loss_rate_upper_bound*veg_2
    model_2.setObjective ( obj_2, GRB.MINIMIZE)  #define obj function and "the sense"

    #Subject to:
    if too_much == False and land_veg_1 == None and land_veg_2==None:
        c1 = model_2.addConstr ( veg_1 + veg_2 >= total_demand, name = "c1" )
        c2 = model_2.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = model_2.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )

    elif too_much==False and land_veg_1 != None and land_veg_2 != None:
        c6 = model_2.addConstr((s3.calc_optimal_input(crop_optimal=veg_2, loss_rates=veg_2_edss['upper_loss_rate']) / veg_2_edss['plants_hact'] + 
        s3.calc_optimal_input(crop_optimal=veg_1, loss_rates=veg_1_edss['upper_loss_rate']) / veg_1_edss['plants_hact'])
        <= ((land_veg_1*veg_1_edss['squeezing_perc'])+(land_veg_2*veg_2_edss['squeezing_perc'])), name='c6')
        c1 = model_2.addConstr ( veg_1 + veg_2 >= total_demand, name = "c1" )
        c2 = model_2.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = model_2.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )
        # c4 = model_2.addConstr((veg_1+veg_2)<=(original_1*1.3+original_2*1.3))



    elif too_much==True:
        c2 = model_2.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = model_2.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )
        # c4 = model_2.addConstr((veg_1+veg_2)==(original_1*1.3+original_2*1.3))
        c6 = model_2.addConstr((s3.calc_optimal_input(crop_optimal=veg_2, loss_rates=veg_2_edss['upper_loss_rate']) / veg_2_edss['plants_hact'] + 
        s3.calc_optimal_input(crop_optimal=veg_1, loss_rates=veg_1_edss['upper_loss_rate']) / veg_1_edss['plants_hact'])
        == ((land_veg_1*veg_1_edss['squeezing_perc'])+(land_veg_2*veg_2_edss['squeezing_perc'])))


    
    model_2.optimize()
    # model_1.write (filename)  write model to a file

    # print ( 'Obj_2 Function Value - Tot_Exp_Loss_Target: %f' % model_2.objVal)

    # Get values of the decision variables

    veg_1_results.append(model_2.X[0])
    veg_2_results.append(model_2.X[1])
    objs.append(model_2.objVal)
    # for e in model_2.getVars () :
    #     print ( "%s: %g" % ( e.varName, e.x))  

    # --------------------------------------------------------- #

    goal_programming_model = Model(name = "Goal Programming - Minimize sum of max percentage deviation ") 
    goal_programming_model.Params.LogToConsole = 0

    veg_1 = goal_programming_model.addVar ( name = "veg_1", vtype = GRB.CONTINUOUS, lb = 0)
    veg_2 = goal_programming_model.addVar ( name = "veg_2", vtype = GRB.CONTINUOUS, lb = 0)
    Q = goal_programming_model.addVar ( name = " Q", vtype = GRB.CONTINUOUS, lb = 0)


    multi_obj_deviations = Q #minMax variable  ( minimize sum of max percentage deviation )
    goal_programming_model.setObjective ( multi_obj_deviations, GRB.MINIMIZE)  #define obj function and "the sense"

    #Subject to:

    # #Previous targets now constraints:
    # c4 = goal_programming_model.addConstr (min_visible_waste_weight *((veg_1_variability* veg_1 + veg_2_variability*veg_2)-model_1.objVal)/model_1.objVal  <= Q , name = "obj_1")
    # c5 = goal_programming_model.addConstr( min_total_input_weight *(( (veg_1_mean_loss_rate_upper_bound* veg_1) + (veg_2_mean_loss_rate_upper_bound* veg_2))-model_2.objVal)/model_2.objVal <= Q, name = "obj_2" )

    if too_much == False and land_veg_1 == None and land_veg_2 == None:
        c1 = goal_programming_model.addConstr ( veg_1 + veg_2 >= total_demand, name = "c1" )
        c2 = goal_programming_model.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = goal_programming_model.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )
        #Non negativity constraint is embed in the lower-bound when the variable has been defined above
        #Previous targets now constraints:
        c4 = goal_programming_model.addConstr (min_visible_waste_weight *((veg_1_variability* veg_1 + veg_2_variability*veg_2)-model_1.objVal)/model_1.objVal  <= Q , name = "obj_1")
        c5 = goal_programming_model.addConstr( min_total_input_weight *(( (veg_1_mean_loss_rate_upper_bound* veg_1) + (veg_2_mean_loss_rate_upper_bound* veg_2))-model_2.objVal)/model_2.objVal <= Q, name = "obj_2" )
        
    elif too_much==False and land_veg_1 != None and land_veg_2 != None:
        c6 = goal_programming_model.addConstr((s3.calc_optimal_input(crop_optimal=veg_2, loss_rates=veg_2_edss['upper_loss_rate']) / veg_2_edss['plants_hact'] + 
        s3.calc_optimal_input(crop_optimal=veg_1, loss_rates=veg_1_edss['upper_loss_rate']) / veg_1_edss['plants_hact'])
        <=  ((land_veg_1*veg_1_edss['squeezing_perc'])+(land_veg_2*veg_2_edss['squeezing_perc'])), name='c6') 
        c1 = goal_programming_model.addConstr ( veg_1 + veg_2 >= total_demand, name = "c1" )
        c2 = goal_programming_model.addConstr ( veg_1  >= veg_1_edss['demand'],  name = "c2" )    
        c3 = goal_programming_model.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )
        c4 = goal_programming_model.addConstr (min_visible_waste_weight *((veg_1_variability* veg_1 + veg_2_variability*veg_2)-model_1.objVal)/model_1.objVal  <= Q , name = "obj_1")
        c5 = goal_programming_model.addConstr( min_total_input_weight *(( (veg_1_mean_loss_rate_upper_bound* veg_1) + (veg_2_mean_loss_rate_upper_bound* veg_2))-model_2.objVal)/model_2.objVal <= Q, name = "obj_2" )
        # c6 = goal_programming_model.addConstr((veg_1+veg_2)<=(original_1*1.3+original_2*1.3))

    elif too_much==True:
        c2 = goal_programming_model.addConstr ( veg_1  >=  veg_1_edss['demand'],  name = "c2" )    
        c3 = goal_programming_model.addConstr (veg_2 >= veg_2_edss['demand'], name = "c3" )
        # c6 = goal_programming_model.addConstr((veg_1+veg_2)==(original_1*1.3+original_2*1.3))
        c8 = goal_programming_model.addConstr((s3.calc_optimal_input(crop_optimal=veg_2, loss_rates=veg_2_edss['upper_loss_rate']) / veg_2_edss['plants_hact'] + 
        s3.calc_optimal_input(crop_optimal=veg_1, loss_rates=veg_1_edss['upper_loss_rate']) / veg_1_edss['plants_hact'])
        == ((land_veg_1*veg_1_edss['squeezing_perc'])+(land_veg_2*veg_2_edss['squeezing_perc'])), name='c8')
        
        # #Previous targets now constraints:
        c4 = goal_programming_model.addConstr (min_visible_waste_weight *((veg_1_variability* veg_1 + veg_2_variability*veg_2)-model_1.objVal)/model_1.objVal  <= Q , name = "obj_1")
        c5 = goal_programming_model.addConstr( min_total_input_weight *(( (veg_1_mean_loss_rate_upper_bound* veg_1) + (veg_2_mean_loss_rate_upper_bound* veg_2))-model_2.objVal)/model_2.objVal <= Q, name = "obj_2" )
        

    
    goal_programming_model.optimize()
    # model_1.write (filename)  write model to a file

    veg_1_results.append(goal_programming_model.X[0])
    veg_2_results.append(goal_programming_model.X[1])
    objs.append(goal_programming_model.objVal)


    # for e in goal_programming_model.getVars () :
    #     print ( "%s: %g" % ( e.varName, e.x))
    
    # print ("veg_1 vector", veg_1_results)   
    # print ("veg_2 vector", veg_2_results) 
    # print ("Obj vector",objs)
    

    # OPTIMIZATION RESULTS:
    optimization_results = {f'{veg_1_edss["name"]}_crop_optimal':math.ceil(veg_1_results [2]), f'{veg_2_edss["name"]}_crop_optimal':math.ceil(veg_2_results [2])}
  
#   # To plot based on input rather than output - discretion of the user which point of view is preferable 
    cropmix_veg_1=s3.calc_optimal_input(crop_optimal=optimization_results[f'{veg_1_edss["name"]}_crop_optimal'], loss_rates=veg_1_edss['upper_loss_rate'])
    cropmix_veg_2=s3.calc_optimal_input(crop_optimal=optimization_results[f'{veg_2_edss["name"]}_crop_optimal'], loss_rates=veg_2_edss['upper_loss_rate'])
    cropmix_original_1 = s3.calc_optimal_input(crop_optimal=original_1, loss_rates=veg_1_edss['upper_loss_rate'])
    cropmix_original_2 =s3.calc_optimal_input(crop_optimal=original_2, loss_rates=veg_2_edss['upper_loss_rate'])
#   #
    # Update total_workers to keep time constant upon random demands updates
    if land_veg_1 != None and land_veg_2!=None and too_much==False or too_much ==True:
        proportion =  ((total_workers_available *(veg_1_results [2]+veg_2_results [2])) / (original_1+original_2))/total_workers_available
        total_workers_available = math.ceil(proportion * total_workers_available)
#   #     
        optimization_results[f'{veg_1_edss["name"]}_workers'] = round(optimization_results[f'{veg_1_edss["name"]}_crop_optimal']*total_workers_available/total_demand)
        optimization_results[f'{veg_2_edss["name"]}_workers'] = total_workers_available - optimization_results[f'{veg_1_edss["name"]}_workers']
        print('OPTIMIZATION RESULT \n', optimization_results,"\n") 

    #------------------------------ PLOT ---------------------------#

    minMax_output_target_1 = veg_1_variability*veg_1_results [2] + veg_2_variability*veg_2_results [2]
    minMax_output_target_2 = veg_1_mean_loss_rate_upper_bound*veg_1_results [2] + veg_2_mean_loss_rate_upper_bound*veg_2_results [2]

    minMax_input_target_1 = veg_1_variability*cropmix_veg_1 + veg_2_variability*cropmix_veg_2
    minMax_input_target_2 = veg_1_mean_loss_rate_upper_bound*cropmix_veg_1 + veg_2_mean_loss_rate_upper_bound*cropmix_veg_2
    
    # PRINT FOR DEBUGGING :
    # print('\n', 'veg 1 last', cropmix_veg_1, 'veg 2 last', cropmix_veg_2, '\n',
    # 'oroginal cropmix 1', cropmix_original_1, 'original cropmix 2', cropmix_original_2, '\n',
    # 'max total = ',(cropmix_original_1*1.3)+(cropmix_original_2*1.3), 'sum last cropmix = ', cropmix_veg_2+cropmix_veg_1, '\n')

    
    
    if too_much == True:

        d = np.linspace(20000,300000,5000)
        x,y = np.meshgrid(d,d)


        plt.figure(figsize=(6,6))
        max_plantable_1 = cropmix_original_1*1.3
        max_plantable_2 = cropmix_original_2*1.3
        max_total = max_plantable_2+max_plantable_1
        total_demand = cropmix_veg_2+cropmix_veg_1

        plt.imshow((x>=cropmix_veg_1) & (y>=cropmix_veg_2) & (x+y<=max_total),
                extent=(x.min()+1,x.max()+1,y.min()+1,y.max()+1),origin="lower", cmap="Greys", alpha = 0.3)  # Feasibile area

        # plt.xlim(45000,max_total*1.1)
        # plt.ylim(45000,max_total*1.1)
        plt.hlines(max(cropmix_veg_2, original_2), 0,max_total, linestyles='--', alpha=0.6, )
        plt.vlines(max(cropmix_veg_1, original_1), 0,max_total, linestyles='--', alpha=0.6)
        plt.plot([0,total_demand],[total_demand,0], linestyle='--', label='total demand', alpha=0.6, c='blue')
        plt.plot([0, max_total], [max_total,0], linestyle='--', label='max plantable', alpha=0.6, c='cyan')
        plt.annotate('Feasable area overlap', xy=(total_demand/2,total_demand/2))

        # minMax plot
        plt.plot([minMax_input_target_1/veg_1_variability,0],[0,minMax_input_target_1/veg_2_variability], "red", label='Model 1', linewidth=1, alpha=0.6) #Solution of model 1
        plt.plot([minMax_input_target_2/veg_1_mean_loss_rate_upper_bound,0], [0,minMax_input_target_2/veg_2_mean_loss_rate_upper_bound], 'red',label='Model 2', lw=1, alpha=0.6)  #Solution of model 2
        
    
    elif land_veg_1 != None and land_veg_2!=None and too_much==False:

        d = np.linspace(20000,300000,5000)
        x,y = np.meshgrid(d,d)

        plt.figure(figsize=(6,6))
        max_plantable_1 = cropmix_original_1*1.3
        max_plantable_2 = cropmix_original_2*1.3
        max_total = max_plantable_2+max_plantable_1
        total_demand = cropmix_veg_2+cropmix_veg_1

        plt.imshow((x>=cropmix_veg_1) & (y>=cropmix_veg_2) & (x+y<=max_total),
                extent=(x.min()+1,x.max()+1,y.min()+1,y.max()+1),origin="lower", cmap="Greys", alpha = 0.3)  # Feasibile area

        # plt.xlim(40000,max_total*1.1)
        # plt.ylim(40000,max_total*1.1)
        plt.hlines(cropmix_veg_2, 0,max_total, linestyles='--', alpha=0.6)
        plt.vlines(cropmix_veg_1, 0,max_total, linestyles='--', alpha=0.6)
        plt.plot([0,total_demand],[total_demand,0], linestyle='--', label='total demand', alpha=0.6, c='blue')
        plt.plot([0, max_total], [max_total,0], linestyle='--', label='max plantable', alpha=0.6, c='cyan')
        plt.annotate('Feasable area', xy=(total_demand/2,total_demand/2))

        # minMax plot
        plt.plot([minMax_input_target_1/veg_1_variability,0],[0,minMax_input_target_1/veg_2_variability], "red", label='Model 1', linewidth=1, alpha=0.6) #Solution of model 1
        plt.plot([minMax_input_target_2/veg_1_mean_loss_rate_upper_bound,0], [0,minMax_input_target_2/veg_2_mean_loss_rate_upper_bound], 'red',label='Model 2', lw=1, alpha=0.6)  #Solution of model 2
        

    else:
        d = np.linspace(1000,100000,5000)
        x,y = np.meshgrid(d,d)
        plt.figure(figsize=(6,6))
        plt.imshow((x>=veg_1_edss['demand']) & (y>=veg_2_edss['demand']) & (x+y>=total_demand), 
                extent=(x.min()+1,x.max()+1,y.min()+1,y.max()+1),origin="lower", cmap="Greys", alpha = 0.3)  # Feasibile area
        plt.xlim(5000,total_demand)
        plt.ylim(5000,total_demand)
        plt.hlines(veg_2_edss['demand'], 0,total_demand, linestyles='--', alpha=0.6)
        plt.vlines(veg_1_edss['demand'], 0,total_demand, linestyles='--', alpha=0.6)
        plt.plot([0,total_demand],[total_demand,0], linestyle='--', label='constraints', alpha=0.6)

        # minMax plot
        plt.plot([minMax_output_target_1/veg_1_variability,0],[0,minMax_output_target_1/veg_2_variability], "red", label='Model 1', linewidth=1, alpha=0.6) #Solution of model 1
        plt.plot([minMax_output_target_2/veg_1_mean_loss_rate_upper_bound,0], [0,minMax_output_target_2/veg_2_mean_loss_rate_upper_bound], 'red',label='Model 2', lw=1, alpha=0.6)  #Solution of model 2

        plt.annotate('Feasable area', xy=(total_demand/2,total_demand/2))

    plt.title(f'Optimization for demand number {n_plot}', fontsize=12)
    plt.legend(fontsize=6.5)
    plt.xlabel('veg_1',size = 10)
    plt.ylabel('veg_2', size=10)
    # plt.show()
    n_plot+=1

    return optimization_results
# # CALCULATE OPTIMAL INPUT GIVEN THE ABOVE CROP MIX DECISION AND THE UPPER BOUND OF THE PRODUCTION YIELD LOSS RATE already_fullED WORST CASE SCENARIO (WCs)
# # The farmer is risk adverse and always plan for the WCS
# # Class "Veggie" stores the Veggies attributes and production data 
# # Class "S3" is the digital twin of the real environment and the continuous growth process and stochastic set of activity is tracked in 4 discrete states
# # alternating a time window of activity where workers are required and a time window of queue where no work is needed on the plant and workers are free.
# # Workers are the resources, divided in two pools and are dedicated. 
# # S3 explores alterantive scenarios to optimize human resources usage during the productions and eventually making them floating.
# # Real_loss_rate is a variable in the S3 method "Farming" that simulate the real world stochasticity
#    # ------------------------------------------------------------------------------------------------------------------- #                   

class veggie():

    # creates attributes for each vegetable
    def __init__(self, name, edss, est_yield):
        self.name = name
        self.edss = edss
        self.est_yield = est_yield
        self.stages_dic = {"greenhouse_activity":0, "transplanting activity" :0, 'fertilizing_activity':0, 'harvested':0}
    
    def new_loss_rate(self):
        # Check for new loss rate with a random generator - e.g. if growth reducing factors increas, thus the loss rate assumed increases
        # loss_question = input('Is loss rate still the same? y/n) 
        if np.random.choice([True, False], p=[0.1,0.9]) == True:
            self.edss['upper_loss_rate'] = self.edss['upper_new_loss_rate']
            return True
        else:
            return False

    # This function is to reduce the farming time when theh pool of resources is moved from one vegetable to the other 

    def tot_farm_time(self, n_workers_initial, n_workers_new, stage):
        # total time obtained doing est_yield divided by activity step (1000) * edss['farming_time'] in each step (i) diveded initial workers (7 or 3)
        total_farming_time_initial = [(i*(self.est_yield/self.edss['activity_step']))/n_workers_initial for i in self.edss['farming_time']]
        # total time obtained doing est_yield divided by activity step (1000) * edss['farming_time'] in each step (i) diveded NEW workers (10)
        total_farming_time_new = [(i*(self.est_yield/self.edss['activity_step']))/n_workers_new for i in self.edss['farming_time']]
        # GEtting percentage change for specific state
        perc_change = abs([1-((new/old)*100) for new, old in zip(total_farming_time_new, total_farming_time_initial)][stage])/100

        # Return a list of new farming time if there were 10 workers in each stage
        return [np.array(self.edss['farming_time'])*(1-perc_change), perc_change]



class s3():

    # Initiating environments, creating istances of two vegetables, calculating yield andn land needed (more in detail in the next few lines)
    def __init__(self,names, edss, tot_workers, tot_demand):
        self.env = simpy.rt.RealtimeEnvironment(factor=wall_clockwise_model, strict=False)

        # creating veggie istances
        self.veg_1 = veggie(name=names[0], edss=edss[0], est_yield=0)
        self.veg_2 = veggie(name=names[1], edss=edss[1], est_yield=0)
        self.tot_workers = tot_workers
        self.tot_demand = tot_demand
        self.veg_1_farmers = simpy.Resource(self.env, capacity=1)
        self.veg_2_farmers = simpy.Resource(self.env, capacity=1)

        # OPTIMIZATION crop mix - getting crop mix results which is the optimal OUTPUT with minimized variability and waste
        self.optimization_result = optimization(self.veg_1.edss, self.veg_2.edss, self.tot_workers, self.tot_demand)

        # Here calculates optimal INPUT starting from optimization optimal output results
        self.veg_1.est_yield = self.calc_optimal_input(self.optimization_result[f'{self.veg_1.name}_crop_optimal'], self.veg_1.edss['upper_loss_rate'])
        self.veg_2.est_yield = self.calc_optimal_input(self.optimization_result[f'{self.veg_2.name}_crop_optimal'], self.veg_2.edss['upper_loss_rate'])

        # Calculate land needed for each vegetables  based on optimal input
        self.land_veg_1 = self.total_land_needed(self.veg_1)
        self.land_veg_2 = self.total_land_needed(self.veg_2)

# Save the original demand because the land bought is based on this, and furher increase in sowing activity will use this value 
        self.original_demand_1 = self.optimization_result[f'{self.veg_1.name.title()}_crop_optimal']
        self.original_demand_2 = self.optimization_result[f'{self.veg_2.name.title()}_crop_optimal']

        # Change the original demand into the edss to what was the optimised OUTPUT (e.g. crop mix) for later reference
        self.veg_1.edss['demand'] = self.optimization_result[f'{self.veg_1.name.title()}_crop_optimal']
        self.veg_2.edss['demand'] = self.optimization_result[f'{self.veg_2.name.title()}_crop_optimal']
        

        # Atlernative scenario resources and envionment
        self.env_fast = simpy.rt.RealtimeEnvironment(factor=0.00001, strict=False, initial_time=0)
        self.veg_1_fast = veggie(name=self.veg_1.name, edss=self.veg_1.edss, est_yield=self.veg_1.est_yield)
        self.veg_2_fast = veggie(name=self.veg_2.name, edss=self.veg_2.edss, est_yield=self.veg_2.est_yield)
        self.farmer_veg_1_fast = simpy.Resource(self.env_fast, capacity=1)
        self.farmer_veg_2_fast = simpy.Resource(self.env_fast, capacity=1)

        # To keep track of stage, how many time the retailer updated the demand, and if land has already been used in full
        self.stage = -1
        self.updated_demand = 0
        self.already_full = False
        self.optimazion_alredy_called = False


    def farming(self, env, veg, farmer, stage, farmtime):

        # List stages to keep track of the stage the vegetable is in
        stages = ['greenhouse_activity', 'transplanting activity','fertilizing_activity','harvested']
        # dictionary created to keep track of REAL production as opposed to what was planned 
        actual_production = {k:0 for k in stages}
        while stage<3:
            # print(f'{veg.name.upper()} starts stage {stage}')
            # print('\nlettuce demand', self.veg_1.edss['demand'], 'cauli demand', self.veg_2.edss['demand'], 'total', self.tot_demand)
            
            # print(veg.name, 'est yield beginning while', veg.est_yield)
            stage+=1
            # print(f'STAGE {stage} of {veg.name} STARTS AT {env.now}')

            # Make sure that demand can only be updated twice
            updated_demand = 0
            while veg.stages_dic[stages[stage]] < veg.est_yield:

            # check if there is new demand or loss rate different or if demand is different in the followint nested if/else blocks
            # only working in the first stage, when edits to crop mix and optimal input are still possible
                if stage==0 and  env!=self.env_fast and self.already_full==False:
                # Sowed variable is to keep track of how many hectars have been farmed
                    sowed = (veg.edss['activity_step']/veg.est_yield)

                #  CHECK FOR UPDATES ON LOSS_RATE, e.g. new pests, new weather forecasts
                    if (veg.edss['upper_loss_rate'] != veg.edss['upper_new_loss_rate']) and veg.new_loss_rate() == True:
                        # print(f'+++++++++++ new loss rate detected for {veg.name.upper()}, optimal input updated +++++++++')
                        veg.est_yield = self.calc_optimal_input(self.optimization_result[f'{veg.name}_crop_optimal'],veg.edss["upper_loss_rate"])-veg.stages_dic[stages[stage]]

            # RANDOMLY UPDATE DEMAND FOR A MAX OF 'UPDATED_DEMAND' TIMES
                    elif np.random.choice([True, False], p=[0.05,0.95]) and updated_demand<4 and self.optimazion_alredy_called==False:
                        updated_demand +=1

#  RANDOLY SELECTING IF RETAILER WANTS MORE OF VEG_1/VEG_2, TOTAL_DEMAND, OR ANY COMBINATIONS OF THESE, RANDOMLY!
# random.choise should be substituted by an user input, e.g. input=('How much is increase in tot_demand? ' ... 100000)
# or it could be constanly uploaded from an excel spreadsheet, but for the sake of demostration of the algorithm it has been implemented with random.choice 
                        stochastic_demand = np.random.choice([veg.edss['demand'], self.tot_demand], size=np.random.randint(1,3), replace=False)
                        o_d_v = veg.edss['demand']
                        o_d_tot = self.tot_demand
                        
                        for i in stochastic_demand:
                            if i == veg.edss['demand']:
                                # print('original demand ', veg.edss['demand'], 'original Total', self.tot_demand)
                                increase = np.random.randint(veg.edss['demand']*0.02,veg.edss['demand']*0.15)
                                veg.edss['demand'] += increase
                                self.tot_demand+=increase
                                print(f'New demand from retailer incoming for {veg.name.upper(), veg.edss["demand"]} \n New Total {self.tot_demand}')
                            else:
                                print('tot demand before update', self.tot_demand)
                                self.tot_demand+=np.random.randint(self.tot_demand*0.02,self.tot_demand*0.15)
                                print(f'New demand from retailer incoming for TOTAL_DEMAND {self.tot_demand}')

# Check which vegatable is the one going through the function
                        if veg.name == self.veg_1.name and self.already_full==False:
# already_full optimization, aware that it could return an error because of impossibility to find feasable area and satisfy one constraing
                            try:
                                self.optimazion_alredy_called = True
                                self.optimization_result = optimization(veg.edss, self.veg_2.edss, self.tot_workers, self.tot_demand, self.land_veg_1, self.land_veg_2, sowed, self.original_demand_1, self.original_demand_2)
                                # veg.est_yield = self.calc_optimal_input(self.optimization_result[f'{veg.name}_crop_optimal'], veg.edss['upper_loss_rate'])-veg.stages_dic[stages[stage]]
                                # print(f'Demand has been updated, stil some room for further squeezing_perc: {self.land_veg_1* - self.total_land_needed(veg)} hectars\n')
                                veg.edss['demand'] = self.optimization_result[veg.name+"_crop_optimal"]
                                self.veg_2.edss['demand'] = self.optimization_result[self.veg_2.edss.name+"_crop_optimal"]
                                self.tot_demand = self.optimization_result[veg.name+"_crop_optimal"] + self.optimization_result[self.veg_2.name+"_crop_optimal"]
                                self.optimazion_alredy_called = False
                            except Exception:
                                self.optimazion_alredy_called = True
                                veg.edss['demand'] = o_d_v
                                self.tot_demand = o_d_tot
                                self.already_full = True
# If the optimiztion can't find a solutions, then the farmer will automatialready_fully plant
                                print('********* too much demand, no feasable solution to satisfy all of it, all veggies are being squeezed ************')
                                self.optimization_result = optimization(veg.edss, self.veg_2.edss, self.tot_workers, self.tot_demand, self.land_veg_1, self.land_veg_2, sowed, self.original_demand_1, self.original_demand_2, too_much=True)
                                veg.edss['demand'] = self.optimization_result[veg.name+"_crop_optimal"]
                                self.veg_2.edss['demand'] = self.optimization_result[self.veg_2.name+"_crop_optimal"]
                                self.tot_demand = self.optimization_result[veg.name+"_crop_optimal"] + self.optimization_result[self.veg_2.name+"_crop_optimal"]
                                self.optimazion_alredy_called = False

                        else:
                            try:
                                self.optimazion_alredy_called = True
                                self.optimization_result = optimization(self.veg_1.edss, veg.edss, self.tot_workers, self.tot_demand, self.land_veg_1, self.land_veg_2, sowed, self.original_demand_1, self.original_demand_2)
                                # veg.est_yield = self.calc_optimal_input(self.optimization_result[f'{veg.name}_crop_optimal'], veg.edss['upper_loss_rate'])-veg.stages_dic[stages[stage]]
                                # print(f'Demand has been updated, stil some room for further squeezing_perc: {self.land_veg_2* - self.total_land_needed(veg)} hectars\n')
                                veg.edss['demand'] = self.optimization_result[veg.name+"_crop_optimal"]
                                self.veg_1.edss['demand'] = self.optimization_result[self.veg_1.name+"_crop_optimal"]
                                self.tot_demand = self.optimization_result[veg.name+"_crop_optimal"] + self.optimization_result[self.veg_1.name+"_crop_optimal"]
                                self.optimazion_alredy_called = False
                            except Exception:
                                self.optimazion_alredy_called = True
                                veg.edss['demand'] = o_d_v
                                self.tot_demand = o_d_tot
                                self.already_full = True
                                print('********* too much demand, no feasable solution for optimization, all veggies are being squeezed, all land used **********\n')
                                self.optimization_result = optimization(self.veg_1.edss, veg.edss, self.tot_workers, self.tot_demand, self.land_veg_1, self.land_veg_2, sowed, self.original_demand_1, self.original_demand_2, too_much=True)
                                
                                # veg['demand'] is now assigned to the result of the last optimization - thus the amount of that vegetable that will be produced
                                self.veg_1.edss['demand'] = self.optimization_result[self.veg_1.name+"_crop_optimal"]
                                veg.edss['demand'] = self.optimization_result[veg.name+"_crop_optimal"]
                                # tot demand is the total that will pbe produced, which is the sum of cropmix for veg_1 + cropmix for veg_2
                                self.tot_demand = self.optimization_result[veg.name+"_crop_optimal"] + self.optimization_result[self.veg_1.name+"_crop_optimal"]
                                self.optimazion_alredy_called = False

                        # print('++++++ Demand has been updated ++++++')
                    veg.est_yield = self.calc_optimal_input(self.optimization_result[f'{veg.name}_crop_optimal'], veg.edss['upper_loss_rate'])

                if veg.est_yield-veg.stages_dic[stages[stage]] < veg.edss['activity_step']:
                    with farmer.request() as req:
                        yield req
                        veg.stages_dic[stages[stage]]+= (veg.est_yield-veg.stages_dic[stages[stage]])+1
                        yield env.timeout(farmtime[stage])
                         
                else:
                    with farmer.request() as req:
                        yield req
                        veg.stages_dic[stages[stage]]+=veg.edss['activity_step']
                        yield env.timeout(farmtime[stage])
            print(f'time {round(env.now, 2)},  {stages[stage]} {str(veg.name).upper()}: ACTIVITY FINISHED!\n')

            # Percentage increase/decrease farming time
            perc_veg_1 = veg.tot_farm_time(n_workers_initial=3, n_workers_new=10, stage=stage)[1]
            perc_veg_2 = veg.tot_farm_time(n_workers_initial=7, n_workers_new=10, stage=stage)[1]

            time_after_activity = env.now

            # # # MOVE POOL OF WORKERS THAT ARE FREE EARLIER 
            # if veg == self.veg_1 and self.veg_1.stages_dic[stages[stage]] >= self.veg_1.est_yield and self.veg_2.stages_dic[stages[stage]]< self.veg_2.est_yield and env != self.env_fast:
            #     self.alternative_scenarios(stage=stage-1, veg=self.veg_2)
            #     print('ALTERNATIVE SCENARIO END')
            #     print('- - -'*30)
            #     # decision = input(f'Idle resources detected. Alternative scenario analysis show time saving of {perc_veg_2*100}% if resourses are moved. Proceed? Yes/No ')
            #     decision = np.random.choice(['no', 'yes'], p=[0.5,0.5])
            #     if decision.lower() in ['yes','y']:
            #         self.veg_2.edss['farming_time'][stage] *= (1-perc_veg_2)
            #         while True:
            #             if self.veg_2.stages_dic[stages[stage]]>=self.veg_2.est_yield or env.now+self.veg_2.edss['farming_time'][stage]>(time_after_activity+self.veg_1.edss['stages_interval'][stage]):
            #                 self.veg_2.edss['farming_time'][stage] /= (1-perc_veg_2)
            #                 print('Pool of resources back to original vegetable')
            #                 break
            #             # already_full tot farm time function so that it will return the % change precicely - this is veg_1 workers moving to veg_2, so 7->10
            #             else:
            #                 yield env.timeout(self.veg_2.edss['farming_time'][stage])
            #         print(f'\nYou have saved {perc_veg_2*100}% on activity time for veg_2')
            #             # In case veg_1 queue finishes while pool of workers is still helping out for veg_2, they are already_fulled back to work on veg_1
                    
            
            # elif veg==self.veg_2 and self.veg_2.stages_dic[stages[stage]] >= self.veg_2.est_yield and self.veg_1.stages_dic[stages[stage]] < self.veg_1.est_yield and env != self.env_fast:
            #     self.alternative_scenarios(stage=stage-1, veg=self.veg_1)
            #     print('ALTERNATIVE SCENARIO END')
            #     print('- - -'*30)
            #     # decision = input(f'Idle resources detected. Alternative scenario shows that moving resources you can save {perc_veg_1*100}% time. Do you want to move? ')
            #     decision = np.random.choice(['no', 'yes'], p=[0.5,0.5])
            #     if decision.lower() in ['yes','y']:
            #         self.veg_1.edss['farming_time'][stage] *= (1-perc_veg_1)
            #         while True:
            #             if self.veg_1.stages_dic[stages[stage]] >= self.veg_1.est_yield or env.now+ self.veg_1.edss['farming_time'][stage]<(time_after_activity+self.veg_2.edss['stages_interval'][stage] ):
            #                 self.veg_1.edss['farming_time'][stage] /= (1-perc_veg_1)
            #                 print('Pool of resources back to original vegetable')
            #                 break
            #             else:
            #                 yield env.timeout(self.veg_1.edss['farming_time'][stage])
            #             # already_full tot farm time function so that it will return the % change precicely - this is veg_2 workers moving to veg_1, so 3->10
            #         print(f'\nYou have saved {perc_veg_1}% on activity time for veg_1')
            #                ## In case veg_2 queue finishes while pool of workers is still helping out for veg_1s, they are already_fulled back to work on veg_2
                    

            # # QUEUE - difference between the queue estimated(scheduled time) and the time that has already elapsed while the pool of a veggie was sent to helo the other veggie production
            yield env.timeout(veg.edss['stages_interval'][stage] - (env.now - time_after_activity))
            # print(f'Queue done! {veg.name, env.now} stage {stage} completed')
            real_loss_rate = np.random.uniform(veg.edss['lower_loss_rate'][stage], veg.edss['upper_loss_rate'][stage])
            actual_production[stages[stage]] = abs((veg.est_yield * (1-veg.edss['upper_loss_rate'][stage])) - veg.est_yield * (1- real_loss_rate))

            if stage>=0 and stage <= 2:
                 # Update Estimated Yield of the production stage
                veg.est_yield = veg.est_yield*(1-veg.edss['upper_loss_rate'][stage])
                # print(f'\n EST YIELD in stage {stage, veg.name} {veg.est_yield} \n')
                print(veg.name, ':' ,veg.stages_dic, sep=' ')
            elif stage == 3 and env != self.env_fast:
                actual_production["actual_harvested_product"] = veg.est_yield - (veg.est_yield * real_loss_rate)
                actual_production[f"VISIBILE_OVERPRODUCTION (WASTE)"] = abs((veg.est_yield*((1-real_loss_rate))) - veg.edss['demand'])
                print (str(veg.name).upper(),'Estimated Optimal Est Yield', veg.stages_dic)
                print('\n Estimated marketable product ', veg.name.title(), ' ', {veg.est_yield*(1-veg.edss['upper_loss_rate'][stage])})
                print (actual_production)
                est_production = pd.DataFrame.from_dict(veg.stages_dic, orient='index')
                actual_production_dataframe = pd.DataFrame.from_dict(actual_production, orient='index')
                pd.concat([est_production, actual_production_dataframe], axis=1).plot.bar(stacked=True, figsize=(8,6), title=f'{veg.name} actual production and estimated yield for WCS')
                plt.legend(['Est production', 'Actual Production'])
                plt.xticks(rotation=0, fontsize=10)




            
 

    def total_land_needed(self, veg):
        return veg.est_yield/veg.edss['plants_hact']

    # def self.calc_optimal_input (veg_1_crop_optimal,veg_2_crop_optimal,veg_1_edss["upper_loss_rate"],veg_2_edss["upper_loss_rate"]):
    @staticmethod
    def calc_optimal_input (crop_optimal, loss_rates):
        gamma = 1 - loss_rates[0]
        for i in loss_rates[1:] :
            gamma = gamma * (1 - i)  
        optimal_input = crop_optimal / gamma
        return optimal_input

    def alternative_scenarios(self, stage, veg):
        print('- - -'*30)
        print("ALTERNATIVE SCENARIO START")
        new_farming_time_veg_2 = self.veg_2.tot_farm_time(n_workers_initial=7, n_workers_new=10, stage=stage)[0]
        new_farming_time_veg_1 = self.veg_1.tot_farm_time(n_workers_initial=3, n_workers_new=10, stage=stage)[0]

        if veg.name == self.veg_2.name:
            self.env_fast.process(self.farming(self.env_fast, self.veg_2_fast, self.farmer_veg_2_fast, self.stage, new_farming_time_veg_2))
        else:
            self.env_fast.process(self.farming(self.env_fast, self.veg_1_fast, self.farmer_veg_1_fast, self.stage, new_farming_time_veg_1))
        self.env_fast.run()
    
    def generate(self):
        self.env.process(self.farming(self.env, self.veg_2, self.veg_2_farmers, self.stage, self.veg_1.edss['farming_time']))
        self.env.process(self.farming(self.env, self.veg_1, self.veg_1_farmers, self.stage, self.veg_2.edss['farming_time']))
        self.env.run()        


# total_workers_available = int(input('how many workers? (only integers please) '))
total_workers_available = 10
# For now only the name is asked as an input, ideally all these data will be imported from an excel sreadsheet and handled using pandas, but i can only submit one file for the assessment!
# name_1 = input('Name of the vegetable 1? ')
# name_2 = input('Name of the vegetable 2? ')
name_1 = 'lettuce'
name_2 = 'cauliflower'
edss_lettuce = {'name':name_1.title(), 'upper_loss_rate' : [0.3,0.4,0.4,0.55], 'lower_loss_rate' : [0.2,0.2,0.3,0.4],'upper_new_loss_rate' : [0.3,0.4,0.4,0.55], 'lower_new_loss_rate' : [0.2,0.2,0.3,0.4],
'demand': np.random.randint(8000, 12000),"activity_step":1000,'stages_interval':[70,50,40,50],'farming_time':[2,2,1,24], 'plants_hact':1500, 'squeezing_perc':1.3}

edss_cauli = {'name':name_2.title(), 'upper_loss_rate' : [0.2,0.2,0.3,0.4], 'lower_loss_rate' : [0.1,0.1,0.05,0.1], 'upper_new_loss_rate' : [0.2,0.2,0.3,0.4],'lower_new_loss_rate' : [0.1,0.1,0.05,0.1],
'demand': np.random.randint(8000, 12000),"activity_step":1000,'stages_interval':[60,100,70,50],'farming_time':[3,1,1.5,18], 'plants_hact':1500, 'squeezing_perc':1.3}

total_demand = np.random.randint(edss_lettuce['demand'] + edss_cauli['demand'], 30000)

# Enter time factor for the main environment here:
wall_clockwise_model = 0.0001

farm = s3(names=[name_1.title(), name_2.title()], edss=[edss_lettuce, edss_cauli], tot_workers=total_workers_available, tot_demand=total_demand)
farm.generate()
plt.show()



