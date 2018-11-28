import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import COMMANDCENTER, SCV, SUPPLYDEPOT, REFINERY, BARRACKS, MORPH_SUPPLYDEPOT_LOWER, MARINE

class SentdeBot(sc2.BotAI):
    async def on_step(self, iteration):

        await self.distribute_workers()
        await self.build_workers() #Build scv
        await self.build_supply() #Build supply to increase the limit of population
        await self.down_supply()
        await self.build_gaz()
        await self.expand()
        await self.build_offensive_structure()
        await self.build_offensive_unit()

    async def build_workers(self):
        for cc in self.units(COMMANDCENTER).ready.noqueue:
            if self.can_afford(SCV):
                await self.do(cc.train(SCV))


    async def build_supply(self):
        if self.supply_left < 3 and not self.already_pending(SUPPLYDEPOT):
            cc_onMap = self.units(COMMANDCENTER).ready
            if (cc_onMap.exists):
                if self.can_afford(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near = cc_onMap.first)

    async def down_supply(self):
        for supp_up in self.units(SUPPLYDEPOT).ready:
            if self.can_afford(MORPH_SUPPLYDEPOT_LOWER):
                await self.do(supp_up(MORPH_SUPPLYDEPOT_LOWER))

    async def build_gaz(self):
        if self.supply_used >= 15:
            for cc in self.units(COMMANDCENTER).ready:
                gaz = self.state.vespene_geyser.closer_than(10.0,cc)
                for vespene in gaz:
                    if not self.can_afford(REFINERY):
                        break
                    worker = self.select_build_worker(vespene.position)
                    if worker is None:
                        break
                    if not self.units(REFINERY).closer_than(1.0, vespene).exists:
                        await self.do(worker.build(REFINERY, vespene))

    async def expand(self):
        if self.units(COMMANDCENTER).amount < 2 and self.can_afford(COMMANDCENTER):
            await self.expand_now(COMMANDCENTER)

    async def build_offensive_structure(self):
        barracks_placement_position = self.main_base_ramp.barracks_correct_placement
        if self.units(SUPPLYDEPOT).ready.exists:
            nb_cc = self.units(COMMANDCENTER).amount
            if self.units(BARRACKS).amount < 4*nb_cc:
                if self.can_afford(BARRACKS):
                    await self.build(BARRACKS, barracks_placement_position)

    async def build_offensive_unit(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE) and self.supply_left > 0:
                await self.do(barrack.train(MARINE))

run_game(maps.get("(2)AcidPlantLE"), [
    Bot(Race.Terran, SentdeBot()),
    Computer(Race.Zerg, Difficulty.Easy)
    ], realtime=False)
