import time

from adt.agent import Agent
from adt.vdesktop import VDesktop


agent = Agent()
app = VDesktop(agent)

app.mainloop()