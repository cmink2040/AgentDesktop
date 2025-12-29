import time

from adt.agent_func import Agent
from adt.vdesktop import VDesktop


agent = Agent()
app = VDesktop(agent)

app.mainloop()