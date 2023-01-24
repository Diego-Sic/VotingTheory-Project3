#Diego Sic
#This code creates the plots for the project 3

import numpy as np
import matplotlib.pyplot as plt
import random
def compute_preference_profile(V,C):
    '''This function will compute the preference of each
    voter in a 2D euclidean space, this will be sort
    in an arrary
    Parameter:
        A range in normal distribtuion of voters
        and a range in normal distribution of Candidates
    Return:
        An array with the order of preferences of each
        voter'''
    p = np.full((V.shape[0], C.shape[0]), -1, dtype = np.int32)
    for i, voter in enumerate(V):
        p[i] = voter_preference(voter, C) #TBD
    return p

def voter_preference(v,C):
    '''This function will order a vallot
    depending on its distance in a 2D - euclidean 
    distance
    Parameter:
        A ballot, a list of positions
    Return:
        An array ordered by lowest distance
        to longest distance with the indices corresponding
        to each position'''
    d = np.full((C.shape[0], 2),-1)
    for i, cand in enumerate(C):
        d[i] = (i, euclid2D(cand, v))

    sorted_indices = d[:,1].argsort()
    d = d[sorted_indices]
    return d[:,0]

def euclid2D(p1,p2):
    '''This function computes the distance
    between to points
    Parameter:
        Two points in a euclidean space
    Return:
        An float describing the distance between
        those points'''
    result = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return result**0.5


def k_borda_winners(profile_preferences, k_factor):
    '''This function will receive a profile with ordinal
    ballots, and it's going to determine the k candidates
    in those ballots with the highest borda score
    Parameter:
        An array describing the preferences of all the voters and
        the k-factor to determine how many candidates we want
    Return:
        A list with the first k-factor winners'''
    borda_scores = np.zeros(profile_preferences.shape[1])
    for elmnt in profile_preferences:
        for i, cand in enumerate(elmnt):
            borda_scores[cand] += ((profile_preferences.shape[1]) - (i+1))

    borda_winner = sorted(range(len(borda_scores)), key=lambda i: borda_scores[i])
    return borda_winner[-k_factor:]


def k_approval_winners(profile_preferences, k_factor):
    '''This function will receive a profile with ordinal
    ballots, and it's going to determine the k candidates
    in those ballots with the approval score
    Parameter:
        An array describing the preferences of all the voters and
        the k-factor to determine how many candidates we want
    Return:
        A list with the first k-approval winners'''
    plurality_scores = np.zeros(profile_preferences.shape[1])
    #This is the num of alternatives
    for elmnt in profile_preferences:
        for i, cand in enumerate(elmnt):
            if i <= (k_factor):

                plurality_scores[cand] += 1

    plurality_winners = sorted(range(len(plurality_scores)), key=lambda i: plurality_scores[i])

    return plurality_winners[-k_factor:]



#Code neeeded to compute CC using the Greedy Algorithm

def winner__round(profile_preferences, k , winners):
    '''This function will determine who is the winner
    in each iteration, taking into consideration the previous
    memebers of the committe to determine which candidates are
    already satisfy
    Parameter:
        An array with ordinal vallots of all the voters,
        the size of the committe that will be used as 
        the amount of appprovals a voter can give. A list
        with all the winners
    Return:
        An number describing the winner of the round'''
    scores = {}
    for i in range(len(profile_preferences[0])):
        scores[i] = supporters(i, profile_preferences, k, winners)

    best_candidate = max(scores, key=scores.get)
    return best_candidate

def winner_not_in_list(winners, order):
    '''This function will check if a voter
    has a candidate in the committe
    Parameter:
        A list with all the previous winners
        and the list with the candidates a voter
        approves
    Return:
        A boolean, if the voter has one of their
        approved candidates in the committe
        will return False. Otherwise, will True'''
    for winner in winners:
        if winner in order:
            return False
    return True

def supporters(i, profile_preferences, k, winners):
    '''This function will count how many supporters
    each candidate has in every round, it will exclude
    all the voters that are already satisfied
    Paratemer:
        An integer describing the candidate "i", 
        an array with oridinal ballots, the size of the 
        committee that will be used to determine
        the amount of approvals a voter has. A list
        with all the previous winners
    Return:
        An integer describing the amount of supporters
        a candidate has in every iteration'''
    supporters = 0
    for j in range(len(profile_preferences)):
        if (i in profile_preferences[j][0:k] 
            and winner_not_in_list(winners, profile_preferences[j][0:k])):
            supporters += 1
    return supporters

def is_satisify(ballot, voter_satisfied):
    '''This function will check if a voter is
    satisfied with at least one candidate in the committee
    Parameter:
        A ballot (A tuple with and index an ordinal order of preferences)
        a list with all the ballots of the voters that are already satisfied
    Return:
        A boolean, True if the candidate is satisfied,
        False if the ballot is not in the list of
        satisfied voters'''
    for voter in voter_satisfied:
        if (ballot == voter).all():
            return True

    return False

def voters__satisfied(profile_preferences, winner, k):
    '''This function will check a profile preferences
    and return a list with all the voters that are satisfied
    with the winner of the round
    Paramter:
        An arrary with ordinal ballots, the winner of 
        the round, and the factor k to determine the
        amount of approvals a voter has.
    Return:
       A list with all the voters that are satisfied
       with the winner of the round'''
    voters__satisfied = []
    for i in range(len(profile_preferences)):
        if winner in profile_preferences[i][0:k]:
            voters__satisfied.append(profile_preferences[i]) 
    return voters__satisfied

def CCAV__greedy(profile_preferences, k):
    '''This function will select a committee of size
    k using approval voting under the rule 
    Chamberlin-Courant. The committe will be computed
    using the Greedy Algorithm. The number of approval
    each voter will be fixed at the size of the committe "k"
    Parameter:
        An array with ordinal ballots of all the voters
    Return:
        A list with all the winners'''
    #Set a list of winners
    winners = []
    #Set a list of voters that are satisfied
    voter_satisfied = []

    while(len(winners) < k): 
        #Calculate the winner of each round
        winner = winner__round(profile_preferences, k, winners)
        #Append that winner in the winners list
        winners.append(winner)
        # Use the method extend to add all the voters that are satisfied
        # by the winner of the round to the list "voter_satisfied"
        voter_satisfied.extend(voters__satisfied(profile_preferences, winner, k))

        #Check how many voters are satisfied
        if len(voter_satisfied) >= len(profile_preferences):
            #If all the voters are satisfied I choose the rest
            #of the Committee randomly
            break
    

    while len(winners) < k:
        #I generate an option from 0 until the last candidate 
        #determine by the length of the ballot of the first voter
        option = random.randint(0,len(profile_preferences[0]-1))
        if option not in winners:
            winners.append(option)
    
    return winners


def plot__winners(winners, cands):
    '''This function will receive a list of winners
    and it's going to return a list that can be ploted
    with 2 inner lists describing the x and y axis of
    those winners.
    Parameter:
        A list with the candidates that won, a list
        describing the position of all the candidates
    Return:
        A 2-D list where the first element is
        the coordinate x and the second element
        is the coordinate y'''
    winners__cordinates = [[],[]]

    for i, elemnt in enumerate(cands):
        if i in winners:
            winners__cordinates[0].append(elemnt[0])
            winners__cordinates[1].append(elemnt[1])
    
    return winners__cordinates

def main():

    n =  1000 #voters
    m = 100 #candidates
    mean = 0
    variance = 2

    for i in range(5,16):
        
        k = i #commite size

        #Creating the voters
        rng = np.random.default_rng(seed = 25)
        voters = rng.normal(mean, variance, (n,2))
        cands = rng.normal(mean,variance,(m,2))

        #====================PLOT===========
        plt.plot(voters[:, 0], voters[:, 1], ".b")
        plt.plot(cands[:, 0], cands[:, 1], ".r")
            
        

        preferences = compute_preference_profile(voters,cands)

        # k_borda = k_borda_winners(preferences, k)
        # k_approval = k_approval_winners(preferences,k)
        k__ccgreddy = CCAV__greedy(preferences, k)

        # k_borda_plots = plot__winners(k_borda, cands)
        # k_approval_plots = plot__winners(k_approval, cands)
        k_ccgreedy_plots = plot__winners(k__ccgreddy, cands)


        description = ["Voters", "Candidates", "k-Approval-Based Chamberlin - Courant"]
        #Plotting the data   
        plt.plot(k_ccgreedy_plots[0], k_ccgreedy_plots[1], 'og')
        plt.title(f"Committe of size {k} - Approval-Based Chamberlin - Courant")
        plt.legend(description)
        fig = plt.gcf()
        fig.savefig(f"Committe of size {k} - Approval-Based Chamberlin - Courant")
        plt.clf()

if __name__ == "__main__":
    main()