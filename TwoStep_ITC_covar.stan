// simultaneous fit model for Two-step Task and Intertemporal Choice Task
// Author: Laura Bustamante 
// Date: August 2021
// The inter-task correlations will be estimated from a covariance matrix 

data {
	// number of subjects
	int NS_all; 
	
	// two-step vars 
	int NS_twostep;
	int MT_twostep;
	int NT_twostep[NS_twostep];
	// stay & repeat first stage choice = 1, switch = 0
	int stay_twostep[NS_twostep,MT_twostep]; 
	// rewarded on previous second stage trial = 1, not rewarded = -1 
	matrix[NS_twostep,MT_twostep] reward_twostep;  
	// common transition on previous trial = 1, uncommon = -1
	matrix[NS_twostep,MT_twostep] transition_twostep; 
	
	// ITC vars 
	int NS_itc;
	int MT_itc;
	int NT_itc[NS_itc];
	int SSchoice_itc[NS_itc,MT_itc];
	matrix[NS_itc,MT_itc] SS_itc;  
	matrix[NS_itc,MT_itc] SS_Delay_itc;  
	matrix[NS_itc,MT_itc] LL_itc;
	matrix[NS_itc,MT_itc] LL_Delay_itc;
}

parameters {
	// group beta_s
	vector[7] beta_ms; 
	// note parameters will be:
	// beta_s[s][1] = intercept_twostep
	// beta_s[s][2] = model_free_twostep
    // beta_s[s][3] = transition_twostep
    // beta_s[s][4] = model_based_twostep
    // beta_s[s][5] = inv_temp_itc
    // beta_s[s][6] = intercept_itc
    // beta_s[s][7] = discount_k_itc
    
	//per subject beta_s
	vector[7] beta_s[NS_all]; 
	
	// beta_s covariance scale
	vector<lower=0>[7] tau;	
	
	// beta_s covariance correlation 
	corr_matrix[7] omega;   
}

transformed parameters {
	// initialize covariance matrix
	matrix[7,7] sigma;
	// define covariance matrix
	sigma = quad_form_diag(omega,tau); 
}

model {
	// prior on covariance matrix
	omega ~ lkj_corr(1);  	
	// prior on inter-subject variability
	tau ~ normal(0,2); 
	// prior on group level parameter mean 
	beta_ms ~ normal(0,1); 
	// loop through participants
	for (s in 1:NS_all) {
	  // sample beta_s for this participant 
	  beta_s[s] ~ multi_normal(beta_ms, sigma);
	  	// if this trial is from the two-step task apply two-step model
		if (NT_twostep[s] > 0) {
			for (tt in 1:NT_twostep[s]) {
				stay_twostep[s,tt] ~ bernoulli_logit(beta_s[s][1] + 
				beta_s[s][2]*reward_twostep[s,tt] + 
				beta_s[s][3]*transition_twostep[s,tt] +
				beta_s[s][4]*reward_twostep[s,tt]*transition_twostep[s,tt]);
				}
			}
		// if this trial is from the ITC task apply ITC model
		if (NT_itc[s] > 0) {
			for (ti in 1:NT_itc[s]) {
				SSchoice_itc[s,ti] ~ bernoulli_logit(beta_s[s][5]*(beta_s[s][6] +
				SS_itc[s,ti]/(1+beta_s[s][7]*SS_Delay_itc[s,ti]) +
				LL_itc[s,ti]/(1+beta_s[s][7]*LL_Delay_itc[s,ti])));
				}
		}
	}
}