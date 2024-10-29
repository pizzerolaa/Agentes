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

class Gambler(ap.Agent):
    # Un agente que apuesta con reglas interesantes
    def setup(self):
        self.wealth = np.random.randint(1, 10)  # Riqueza inicial aleatoria entre 1 y 10
    
    def bet_wealth(self):
        if self.wealth > 1:  # Solo apuesta si tiene suficiente riqueza
            # Ajustar probabilidad de ganar según riqueza
            win_probability = 0.5 + (0.1 if self.wealth > 5 else -0.1)  # Ricos tienen ligera ventaja
            # Apuesta un porcentaje de su riqueza actual
            bet_amount = max(1, int(self.wealth * 0.2))  # 20% de la riqueza, mínimo 1
            win = np.random.rand() < win_probability  # Determinar resultado de la apuesta
            if win:
                self.wealth += bet_amount  # Gana la apuesta, incrementa su riqueza
            else:
                self.wealth -= bet_amount  # Pierde la apuesta, disminuye su riqueza
            # No permitir riqueza negativa
            self.wealth = max(0, self.wealth)

class WealthModel(ap.Model):
    # Modelo simple de transferencias, donaciones y apuestas de riqueza
    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)
        self.gamblers = ap.AgentList(self, self.p.gamblers, Gambler)
    
    def step(self):
        self.agents.wealth_transfer()
        self.gamblers.bet_wealth()
    
    def update(self):
        # Registrar el coeficiente de Gini de los agentes WealthAgent y Gambler
        self.record('Gini Coefficient (WealthAgent)', gini([a.wealth for a in self.agents]))
        self.record('Gini Coefficient (Gambler)', gini([g.wealth for g in self.gamblers]))
    
    def end(self):
        # Registrar la riqueza final de cada tipo de agente
        self.agents.record('wealth')
        self.gamblers.record('wealth')

parameters = {
    'agents': 100,
    'gamblers': 10,
    'steps': 100,
}

model = WealthModel(parameters)
results = model.run()

data = results.variables.WealthModel
ax = data.plot()

results.variables.Gambler["name"] = "Gambler"
results.variables.WealthAgent["name"] = "WealthAgent"
df = pd.concat([results.variables.Gambler, results.variables.WealthAgent])

sns.histplot(data=df, binwidth=1, x="wealth", hue="name", kde=False)
plt.show()
