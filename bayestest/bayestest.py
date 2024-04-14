import numpy as np
import pymc as pm
from tabulate import tabulate
import pandas as pd

class Variant:
    def __init__(self, name, visitors, successes, revenue=None, control=False):
        self.name = name
        self.visitors = visitors
        self.successes = successes
        self.control = control
        self.revenue = revenue  # Initialize without revenue data
        self.trace = None
        self.model = None

    def add_revenue(self, total_revenue):
        self.revenue = total_revenue

class bayesTest:
    def __init__(self):
        self.variants = []
        #self.model = None
        self.trace = None
        self.alpha = 1  # Default alpha for Beta prior
        self.beta = 1   # Default beta for Beta prior
        self.revenue_alpha = None
        self.revenue_test = False
        self.results = None

    def conversion_rate_prior(self, alpha=1, beta=1, plot=False):
        self.alpha = alpha
        self.beta = beta
        return
        
    def revenue_prior(self, alpha=1.5, beta=28, plot=False):
        self.revenue_alpha = alpha
        self.revenue_beta = beta
        self.revenue_test = True
        return 
  

    def add(self, visitors, successes, revenue=None, name=None, control=False):
        variant = Variant(name, visitors, successes, revenue, control)
        self.variants.append(variant)

    def infer(self, samples=5000):
        
       #if not self.revenue_test:
        if False:
            # Sample Conversion Rate posterior
            for variant in self.variants:
                with pm.Model():
                    # Copied from PyMC directly:
                    p = pm.Beta("p", alpha=self.alpha, beta=self.beta)
                    y = pm.Binomial("y", n=variant.visitors, p=p, observed=variant.successes)
                    trace = pm.sample(draws=samples, step=pm.Slice())
                    variant.trace = trace
                    
        elif True:
            for variant in self.variants:
                
                    with pm.Model() as model:
                        theta = pm.Beta(
                            "theta",
                            alpha=self.alpha,
                            beta=self.beta,
                            #shape=num_variants,
                        )
                        converted = pm.Binomial(
                            "converted", n=variant.visitors, p=theta, observed=variant.successes, #shape=num_variants
                        )
                        if self.revenue_test:
                            
                            lam = pm.Gamma(
                                "lam",
                                alpha=self.revenue_alpha,
                                beta=self.revenue_beta,
                                #shape=num_variants,
                            )
                            revenue = pm.Gamma(
                                "revenue", alpha=variant.visitors, beta=lam, observed=variant.revenue, #shape=num_variants
                            )
                            revenue_per_visitor = pm.Deterministic("revenue_per_visitor", theta * (1 / lam))
                        trace = pm.sample(draws=samples, chains=2, tune=750, step=pm.Slice())
                        variant.trace = trace
                                

    def summary(self):
        
        all_cr_simulations = [] # conversion rate simulations
        if self.revenue_test:
            all_rev_simulations = [] # revenue per visitor simulations
        
        for variant in self.variants:
           #all_simulations.append(variant.trace.posterior.p.values.flatten())
           all_cr_simulations.append(variant.trace.posterior.theta.values.flatten())
           if self.revenue_test:
               all_rev_simulations.append(variant.trace.posterior.revenue_per_visitor.values.flatten())
        
        # Stack the posterior samples into a single NumPy array for efficient computation
        # The shape of the stacked array would be (num_samples, num_variants)
        posterior_stacked = np.stack(all_cr_simulations, axis=1)
        # Identify the index of the variant with the highest value for each sample
        winning_variants = np.argmax(posterior_stacked, axis=1)
        # Calculate the winning probability for each variant
        # This computes how often each variant wins, i.e., has the highest posterior sample value
        winning_probabilities = np.mean(winning_variants == np.arange(posterior_stacked.shape[1])[:, None], axis=1)
        
        summary_df = pd.DataFrame({'Variant': [variant.name for variant in self.variants],
                          'Prob. Winner CR': [f'{x:.2%}' for x in winning_probabilities],
                          })
        
        if self.revenue_test:
            posterior_revenue_stacked = np.stack(all_rev_simulations, axis=1)
            winning_variants_rev = np.argmax(posterior_revenue_stacked, axis=1)
            winning_probabilities_rev = np.mean(winning_variants_rev == np.arange(posterior_revenue_stacked.shape[1])[:, None], axis=1)
        
            summary_df['Prob. Winner Rev'] = [f'{x:.2%}' for x in winning_probabilities_rev]
        

        
            
                


            
        print(tabulate(summary_df, headers='keys', tablefmt='grid'))