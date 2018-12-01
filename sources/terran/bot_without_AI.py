import sc2, random
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.ids.unit_typeid import UnitTypeId

class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.MAX_SCV = 60
        self.scv_scouting = 0
        self.iteration_per_min = 165
    async def on_step(self, iteration):
        self.iteration=iteration
        await self.distribute_workers()
        await self.build_workers() #Build scv
        await self.build_supply() #Build supply to increase the limit of population
        await self.morph_cc_in_orbital()
        await self.drop_mule()
        await self.down_supply()
        await self.build_gaz()
        await self.scout()
        await self.expand()
        await self.build_offensive_structure()
        await self.build_offensive_unit()
        await self.defend()
        await self.attack()

    async def build_workers(self):
        if self.units(SCV).amount < self.MAX_SCV:
            for cc in self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).ready.noqueue:
                if self.can_afford(SCV) and not self.can_afford(ORBITALCOMMAND):
                    await self.do(cc.train(SCV))

    async def drop_mule(self):
        for oc in self.units(ORBITALCOMMAND).ready:
            abilities = await self.get_available_abilities(oc)
            if CALLDOWNMULE_CALLDOWNMULE in abilities:
                for cc_alt in self.units(ORBITALCOMMAND).ready:
                    pack_minerals = self.state.mineral_field.closer_than(10.0,cc_alt)
                    for minerals in pack_minerals:
                        await self.do(oc(CALLDOWNMULE_CALLDOWNMULE,minerals))


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

    async def morph_cc_in_orbital(self):
        for cc in self.units(COMMANDCENTER).ready:
            if self.can_afford(ORBITALCOMMAND):
                await self.do(cc(UPGRADETOORBITAL_ORBITALCOMMAND))

    async def build_gaz(self):
        if self.supply_used >= 15:
            if self.units(REFINERY).amount <1 or (self.units(REFINERY).amount >= 1 and self.units(BARRACKS).amount >= 1):
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

    async def scout(self):
        if self.units(SCV).amount >= 15 and self.scv_scouting < 1:
            self.scv_scouting=1;
            scv_scout = self.units(SCV)[0]
            enemy_location = self.known_enemy_structures.random_or(self.enemy_start_locations[0]).position
            await self.do(scv_scout.move(enemy_location))

    async def expand(self):
        if self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).amount < 3 and self.can_afford(COMMANDCENTER) and self.units(BARRACKS).amount >= 1 :
            await self.expand_now(COMMANDCENTER)

    async def build_offensive_structure(self):
        barracks_placement_position = self.main_base_ramp.barracks_correct_placement
        if self.units.of_type([SUPPLYDEPOT, SUPPLYDEPOTLOWERED, SUPPLYDEPOTDROP]).ready.exists:
            nb_cc = self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).amount
            if self.units(BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) <= 0:
                if self.can_afford(BARRACKS):
                    await self.build(BARRACKS, barracks_placement_position)
            elif (self.units(BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 4*nb_cc) and self.units(COMMANDCENTER).amount > 1:
                if self.can_afford(BARRACKS):
                    await self.build(BARRACKS, barracks_placement_position)

    async def build_offensive_unit(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE) and self.supply_left > 0:
                await self.do(barrack.train(MARINE))

    async def defend(self):
        if self.units(MARINE).idle.amount >= 5:
            for struct in self.units.of_type([SUPPLYDEPOT, SUPPLYDEPOTLOWERED, SUPPLYDEPOTDROP,BARRACKS, COMMANDCENTER, REFINERY]).ready:
                for enemy in self.known_enemy_units.not_structure:
                    if enemy.position.to2.distance_to(struct.position.to2) < 10:
                        for ma in self.units(MARINE).idle:
                            await self.do(ma.attack(enemy))


    async def attack(self):
        if self.units(MARINE).idle.amount >= 15:
            for ma in self.units(MARINE).idle:
                await self.do(ma.attack(self.enemy_start_locations[0]))


run_game(maps.get("(2)AcidPlantLE"), [
    Bot(Race.Terran, SentdeBot()),
    Computer(Race.Zerg, Difficulty.Medium)
    ], realtime=False)
