import agentpy as ap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class WealthAgent(ap.Agent):
    # Un agente con riqueza
    def setup(self):
        self.wealth = 1
    
    def wealth_transfer(self):
        if self.wealth > 0:
            partner = self.model.agents.random()
            partner.wealth += 1
            self.wealth -= 1

def gini(x):
    """Calcular el coeficiente de Gini"""
    x = np.array(x)
    mad = np.abs(np.subtract.outer(x, x)).mean()  # Diferencia absoluta media
    rmad = mad / np.mean(x)  # Diferencia absoluta media relativa
    return 0.5 * rmad

class Philanthropist(ap.Agent):
    # Un agente que dona a otros con menos riqueza
    def setup(self):
        self.wealth = 5  # Un nivel de riqueza inicial alto
    
    def donate_wealth(self):
        if self.wealth > 1:  # Donar solo si tiene suficiente riqueza
            # Encuentra un agente aleatorio con menor riqueza
            partner = self.model.agents.select(self.wealth > self.model.agents.wealth).random()
            if partner:
                partner.wealth += 1
                self.wealth -= 1

class WealthModel(ap.Model):
    # Modelo simple de transferencias y donaciones de riqueza
    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)
        self.philanthropists = ap.AgentList(self, self.p.philanthropists, Philanthropist)
    
    def step(self):
        self.agents.wealth_transfer()
        self.philanthropists.donate_wealth()
    
    def update(self):
        # Registrar el coeficiente de Gini de los agentes de Wealth y de los Philanthropist
        self.record('Gini Coefficient (WealthAgent)', gini([a.wealth for a in self.agents]))
        self.record('Gini Coefficient (Philanthropist)', gini([p.wealth for p in self.philanthropists]))
    
    def end(self):
        # Registrar la riqueza final de cada tipo de agente
        self.agents.record('wealth')
        self.philanthropists.record('wealth')

parameters = {
    'agents': 100,
    'philanthropists': 10,
    'steps': 100,
}

model = WealthModel(parameters)
results = model.run()

data = results.variables.WealthModel
ax = data.plot()

results.variables.Philanthropist["name"] = "Philanthropist"
results.variables.WealthAgent["name"] = "WealthAgent"
df = pd.concat([results.variables.Philanthropist, results.variables.WealthAgent])

sns.histplot(data=df, binwidth=1, x="wealth", hue="name", kde=False)
plt.show()
