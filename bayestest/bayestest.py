import numpy as np
import pymc as pm

class Variant:
    def __init__(self, name, visitors, successes, revenue=None, control=False):
        self.name = name
        self.visitors = visitors
        self.successes = successes
        self.control = control
        self.revenue = revenue  # Initialize without revenue data

    def add_revenue(self, total_revenue):
        self.revenue = total_revenue

class bayesTest:
    def __init__(self):
        self.variants = []
        self.model = None
        self.trace = None
        self.alpha = 1  # Default alpha for Beta prior
        self.beta = 1   # Default beta for Beta prior
        self.revenue_alpha = None
        self.revenue = None
        self.results = None

    def prior(self, alpha=1, beta=1, revenue=False, plot=False):
        if revenue:
            self.revenue_alpha = alpha
            self.revenue_beta = beta
        else:
            self.alpha = alpha
            self.beta = beta
        
        if plot:
            # plot
            pass

    def add(self, visitors, successes, revenue=None, name=None, control=False):
        variant = Variant(name, visitors, successes, revenue, control)
        self.variants.append(variant)

    def infer(self, samples=10000):
        with pm.Model() as self.model:
            # Lists to store model variables for conversion rates and revenues
            theta = []  # List to store theta for each variant's conversion rate
            lambda_ = []  # List to store lambda for each variant's revenue (if applicable)

            for variant in self.variants:
                # Update alpha and beta based on observed data for conversion rate
                alpha_prime = self.alpha + variant.successes
                beta_prime = self.beta + (variant.visitors - variant.successes)
                
                # Model conversion rate using a Beta distribution
                theta.append(pm.Beta(f'theta_{variant.name}', alpha=alpha_prime, beta=beta_prime))

                # If revenue data is available and there are successful conversions, model revenue
                if variant.revenue is not None and variant.successes > 0:
                    avg_revenue = variant.revenue / variant.successes

                    # Use specified priors for the Gamma distribution
                    # Ensure `self.revenue_alpha` and `self.revenue_beta` are defined and accessible
                    # You might adjust the calculation or usage of `self.revenue_alpha` and `self.revenue_beta`
                    # based on your understanding of the data and desired model behavior
                    lambda_.append(pm.Gamma(f'lambda_{variant.name}', 
                                            alpha=self.revenue_alpha, 
                                            beta=self.revenue_beta, 
                                            observed=avg_revenue))

            # Perform inference
            self.trace = pm.sample(samples, return_inferencedata=True)


    def samples(self):
        # Debug: Print available variables in the posterior
        print("Available variables in the posterior:", self.trace.posterior.data_vars)
        
        conversion_samples = {
            variant.name: self.trace.posterior[f'theta_{variant.name}'].values.flatten()
            for variant in self.variants
            if f'theta_{variant.name}' in self.trace.posterior.data_vars  # Ensure variable is present
        }
        
        # Initialize revenue_samples with None for all variants
        revenue_samples = {variant.name: None for variant in self.variants}
        
        # Update revenue_samples only for variants with revenue data in the posterior
        for variant in self.variants:
            revenue_var_name = f'lambda_{variant.name}'
            if revenue_var_name in self.trace.posterior.data_vars:  # Check if the revenue variable is present
                revenue_samples[variant.name] = self.trace.posterior[revenue_var_name].values.flatten()
            else:
                print(f"Revenue data not found in the posterior for variant: {variant.name}")  # Debug message

        return conversion_samples, revenue_samples

    
    def summary(self):
        conversion_samples, revenue_samples = self.samples()

        # Number of variants for normalization
        num_variants = len(conversion_samples)
        
        # Initialize dictionary to store win counts for each variant
        conversion_win_counts = {name: 0 for name in conversion_samples}
        
        # For each sample, determine which variant had the highest conversion rate
        for i in range(len(next(iter(conversion_samples.values())))):
            max_conversion_rate = float('-inf')
            winning_variant = None
            for name, samples in conversion_samples.items():
                if samples[i] > max_conversion_rate:
                    max_conversion_rate = samples[i]
                    winning_variant = name
            if winning_variant:
                conversion_win_counts[winning_variant] += 1

        # Calculate the probability of being the winner for each variant
        conversion_win_probs = {name: count / len(next(iter(conversion_samples.values()))) for name, count in conversion_win_counts.items()}

        # Initialize win counts for revenue probabilities
        revenue_win_counts = {name: 0 for name in revenue_samples if revenue_samples[name] is not None}
        num_revenue_comparisons = len(revenue_win_counts)

        if num_revenue_comparisons > 0:  # Proceed only if there are variants with revenue data
            for i in range(len(next(iter(revenue_samples.values())))):
                max_revenue = float('-inf')
                winning_variant_revenue = None
                for name, samples in revenue_samples.items():
                    if samples is not None and samples[i] > max_revenue:
                        max_revenue = samples[i]
                        winning_variant_revenue = name
                if winning_variant_revenue:
                    revenue_win_counts[winning_variant_revenue] += 1

            # Calculate the probability of being the winner for each variant with revenue data
            revenue_win_probs = {name: count / len(next(iter(revenue_samples.values()))) for name, count in revenue_win_counts.items()}
        else:
            revenue_win_probs = {name: 'N/A' for name in conversion_samples}

           # Print the summary table header
        print("| Variant | Probability of Winning on Conversions | Probability of Winning on Revenue |")
        print("|---------|----------------------------------------|-----------------------------------|")
        
        # Iterate through all variants to print their win probabilities for both conversions and revenue
        for name in conversion_samples:
            # Fetch the conversion win probability, default to 'N/A' if not found (should not happen in practice)
            conv_prob = conversion_win_probs.get(name, 'N/A')  
            
            # Fetch the revenue win probability, use 'N/A' for variants without revenue data or comparisons
            rev_prob = revenue_win_probs.get(name, 'N/A') 
            
            # Format the revenue probability; if it's a number, format to two decimal places, otherwise, directly use 'N/A'
            rev_prob_formatted = f"{rev_prob:.2f}" if rev_prob != 'N/A' else rev_prob
            
            # Print the summary line for the current variant
            print(f"| {name} | {conv_prob:.2f} | {rev_prob_formatted} |")
