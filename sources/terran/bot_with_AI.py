import sc2, random
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer, Human
from sc2.constants import *
import cv2
import numpy as np

class SentdeBot(sc2.BotAI):
    def __init__(self):
        self.MAX_SCV = 60
        self.scv_scouting = 0
        self.iteration_per_min = 165
    async def on_step(self, iteration):
        self.iteration=iteration
        await self.outputRGB()
        await self.distribute_workers()
        await self.build_workers() #Build scv
        await self.build_supply() #Build supply to increase the limit of population
        await self.morph_cc_in_orbital()
        await self.drop_mule()
        await self.down_supply()
        #await self.build_gaz()
        await self.scout()
        await self.expand()
        await self.build_offensive_structure()
        await self.build_offensive_unit()
        await self.defend()
        await self.attack()

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to

    async def scout(self):
        if len(self.units(SCV)) > 0:
            scout = self.units(SCV)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                print(move_to)
                await self.do(scout.move(move_to))

    async def outputRGB(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        #Exemple -> COMMANDCENTER : 3 = sight, (0,255,0) = RGB
        sight_def = {
                        COMMANDCENTER: [11, (0, 255, 0)],
                        ORBITALCOMMAND: [11, (0, 255, 0)],
                        SUPPLYDEPOT: [9, (20, 235, 0)],
                        SUPPLYDEPOTLOWERED: [9, (20, 235, 0)],
                        SCV: [8, (55, 200, 0)],
                        MARINE: [9, (55, 200, 0)],
                        REFINERY: [9, (55, 200, 0)],
                        BARRACKS: [9, (200, 100, 0)],
                    }

        #Exemple -> COMMANDCENTER : 3 = size, (0,255,0) = RGB
        size_def = {
                        COMMANDCENTER: [3, (0, 255, 0)],
                        ORBITALCOMMAND: [3, (0, 255, 0)],
                        SUPPLYDEPOT: [1, (20, 235, 0)],
                        SUPPLYDEPOTLOWERED: [1, (20, 235, 0)],
                        SCV: [1, (55, 200, 0)],
                        MARINE: [1, (55, 200, 0)],
                        REFINERY: [1, (55, 200, 0)],
                        BARRACKS: [2, (200, 100, 0)],
                    }

        for unit_sight in sight_def:
            print(unit_sight)
            for i in self.units(unit_sight).ready:
                pos = i.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), sight_def[unit_sight][0],(255,250,250), -1)

        for unit_size in size_def:
            print(unit_size)
            for i in self.units(unit_size).ready:
                print(i)
                pos = i.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), size_def[unit_size][0], size_def[unit_size][1], -1)

        for cc in self.units.of_type([COMMANDCENTER, ORBITALCOMMAND]).ready:
            nearby_minerals = self.state.mineral_field.closer_than(10, cc)
            for mineral in nearby_minerals:
                print(mineral)
                pos = mineral.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255,0,0), -1)

        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 2, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units.not_structure:
            pos = enemy_unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (0, 0, 255), -1)

        flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)

        cv2.imshow('OutputRGB', resized)
        cv2.waitKey(1)

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
            elif (self.units(BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 4*nb_cc) and self.units(COMMANDCENTER).amount >= 1:
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
    #Human(Race.Terran),
    Bot(Race.Terran, SentdeBot()),
    Computer(Race.Zerg, Difficulty.Hard)
    ], realtime=False)
