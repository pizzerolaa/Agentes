import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt

#visualization
import seaborn as sns
#more graphics
import pandas as pd

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

parameters = {
    'agents': 100,
    'steps': 100,
    'seed': 42,
}

model = WealthModel(parameters)
results = model.run()
print(results)

data = results.variables.WealthModel
ax = data.plot()
plt.show()