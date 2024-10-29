import agentpy as ap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class WealthAgent(ap.Agent):
    #an agent with wealth
    def setup(self):
        self.wealth = 1
    
    def wealth_transfer(self):
        if self.wealth > 0:
            partner = self.model.agents.random()
            partner.wealth += 1
            self.wealth -= 1

def gini(x):
    """Calculate gini coefficient"""
    x = np.array(x)
    mad = np.abs(np.subtract.outer(x, x)).mean() # Mean absolute difference
    rmad = mad / np.mean(x) #relative mean absolute differente 
    return 0.5 * rmad

class WealthModel(ap.Model):
    #simple model of random wealth transfers
    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)
    
    def step(self):
        self.agents.wealth_transfer()

    def update(self):
        self.record('Gini Coefficient', gini(self.agents.wealth))

    def end(self):
        self.agents.record('wealth')

class RobinHood(ap.Agent):
    def setup(self):
        self.wealth = 0 #robin pobre
    
    def wealth_taking(self):
        """Metodo toma riqueza de otros metodos"""
        partner = self.model.agents.random().to_list()[0]
        partner_wealth = partner.wealth

        if partner.wealth > 3:
            #si su riqueza es mayor a 3, le quitamos su riqueza
            partner.wealth -= (partner_wealth-1)
            self.wealth += (partner_wealth-1)

class WealthModel(ap.Model):
    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)

        self.robin_agents = ap.AgentList(self, self.p.robin_agents, RobinHood)        
    
    def step(self):
        self.agents.wealth_transfer()

        self.robin_agents.wealth_taking()
    
    def update(self):
        self.record('Gini Coefficient (WealthAgent)', gini(self.agents.wealth))
        self.record('Gini Coefficient (RobinAgent)', gini(self.robin_agents.wealth))
    
    def end(self):
        self.agents.record('wealth')
        self.robin_agents.record('wealth')

parameters = {
    'agents': 100,
    'robin_agents': 100,
    'steps': 100,
}

model = WealthModel(parameters)
results = model.run()

data = results.variables.WealthModel
ax = data.plot()

results.variables.RobinHood["name"] = "RobinHood"
results.variables.WealthAgent["name"] = "WealthAgent"

df = pd.concat([results.variables.RobinHood, results.variables.WealthAgent])

sns.histplot(data= df, binwidth= 1, x= "wealth", hue= "name", kde= False)

plt.show()
    