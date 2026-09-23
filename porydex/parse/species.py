from collections import defaultdict
from dataclasses import dataclass

import pathlib
import re

from pycparser.c_ast import Constant, Decl, ExprList, NamedInitializer
from yaspin import yaspin

from porydex.common import name_key
from porydex.model import ExpansionEvoMethod, DAMAGE_TYPE, EGG_GROUP, BODY_COLOR, EVO_METHOD
from porydex.parse import load_data, extract_id, extract_int, extract_u8_str

EXPANSION_GEN9_START = 1289
VANILLA_GEN9_START = 906
EXPANSION_GEN9_OFFSET = EXPANSION_GEN9_START - VANILLA_GEN9_START

@dataclass
class CosmeticFormes:
    base: str
    alts: list[str] | None
    exclude_pattern: str | None

COSMETIC_FORME_SPECIES: dict[str, CosmeticFormes] = {
    'Unown': CosmeticFormes('A', [], None),
    'Vivillon': CosmeticFormes('Meadow', [], None), # Technically, Fancy and Poke Ball are alt forms and not cosmetic, but eff that
    'Furfrou': CosmeticFormes('Natural', [], None),
    'Spewpa': CosmeticFormes('Icy-Snow', None, None),
    'Scatterbug': CosmeticFormes('Icy-Snow', None, None),
    'Burmy': CosmeticFormes('Plant', [], None),
    'Mothim': CosmeticFormes('Plant', None, None),
    'Shellos': CosmeticFormes('West', [], None),
    'Gastrodon': CosmeticFormes('West', [], None),
    'Deerling': CosmeticFormes('Spring', [], None),
    'Sawsbuck': CosmeticFormes('Spring', [], None),
    'Flabébé': CosmeticFormes('Red', [], None),
    'Floette': CosmeticFormes('Red', ['Eternal'], None),
    'Florges': CosmeticFormes('Red', [], None),
    'Tatsugiri': CosmeticFormes('Curly', [], None),
    'Minior': CosmeticFormes('Red', ['Meteor'], r'Meteor-*'),
    'Alcremie': CosmeticFormes('Vanilla-Cream', ['Gmax'], None),
}

@dataclass
class SpecialAbilities:
    form: str
    ability: str

SPECIAL_ABILITIES: dict[str, SpecialAbilities] = {
    'Greninja': SpecialAbilities('Bond', 'Battle Bond'),
    'Zygarde': SpecialAbilities('Power-Construct', 'Power Construct'),
}


# Map named Expansion type constants (TYPE_FIRE, TYPE_WATER, etc.) to the
# indexes used by Porydex's DAMAGE_TYPE table.
TYPE_ID_TO_INDEX = {
    f'TYPE_{name.upper()}': idx
    for idx, name in enumerate(DAMAGE_TYPE)
}


def extract_known_int(expr, known_ids: dict[str, int]) -> int:
    """Read either a numeric C expression or a known named constant."""
    try:
        return extract_int(expr)
    except (AttributeError, ValueError):
        key = extract_id(expr)
        if key in known_ids:
            return known_ids[key]
        raise


def extract_damage_type(expr) -> str:
    """
    Convert an Expansion type expression into the Showdown/Porydex type name.

    Older Porydex DAMAGE_TYPE tables may stop at Fairy while Expansion 1.13
    can expose TYPE_STELLAR as numeric id 20. Handle that here instead of
    indexing past the end of DAMAGE_TYPE.
    """
    try:
        type_id = extract_int(expr)
    except (AttributeError, ValueError):
        type_name = extract_id(expr)

        if type_name in TYPE_ID_TO_INDEX:
            return DAMAGE_TYPE[TYPE_ID_TO_INDEX[type_name]]

        if type_name == 'TYPE_STELLAR':
            return 'Stellar'

        raise ValueError(f'unknown Pokemon type constant: {type_name}')

    if 0 <= type_id < len(DAMAGE_TYPE):
        return DAMAGE_TYPE[type_id]

    # Expansion's Stellar type. This is the value that causes older Porydex
    # versions to throw: IndexError: list index out of range.
    if type_id == 20:
        return 'Stellar'

    raise ValueError(
        f'Pokemon type id {type_id} is not supported by this Porydex '
        f'(DAMAGE_TYPE has {len(DAMAGE_TYPE)} entries)'
    )



def build_form_table_species_ids(species_data) -> dict[str, list[int]]:
    """
    Build each form table's species membership from the CURRENT gSpeciesInfo.

    This makes form handling follow the species IDs in the Expansion project
    being parsed instead of assuming the numeric IDs from another Expansion
    version or an older Porydex cache.
    """
    refs: dict[str, list[int]] = defaultdict(list)

    for species_init in species_data:
        try:
            species_id = extract_int(species_init.name[0])
        except (AttributeError, ValueError, TypeError, IndexError):
            continue

        fields = getattr(getattr(species_init, 'expr', None), 'exprs', None)
        if not fields:
            continue

        for field_init in fields:
            try:
                field_name = field_init.name[0].name
            except (AttributeError, TypeError, IndexError):
                continue

            if field_name != 'formSpeciesIdTable':
                continue

            try:
                table_name = extract_id(field_init.expr)
            except (AttributeError, ValueError):
                break

            if species_id not in refs[table_name]:
                refs[table_name].append(species_id)
            break

    return dict(refs)


def resolve_dynamic_form_name(mon: dict,
                              table_name: str,
                              table: dict[int, str],
                              form_table_species_ids: dict[str, list[int]]) -> tuple[bool, str | None]:
    """
    Resolve a form against the CURRENT species file.

    Returns:
        (is_base_form, form_name)

    form_name is None for the base form or when no safe mapping can be made.
    """
    if not table:
        return False, None

    mon_num = mon['num']
    current_ids = form_table_species_ids.get(table_name, [])
    table_values = list(table.values())

    # Preferred path: use the order of species that reference this table in
    # the currently loaded gSpeciesInfo. That keeps the mapping valid when
    # custom species/forms shift numeric species IDs.
    if current_ids and mon_num in current_ids:
        form_index = current_ids.index(mon_num)

        if form_index == 0:
            return True, None

        if form_index < len(table_values):
            return False, table_values[form_index]

        # Some Expansion layouts put Ogerpon Tera forms after the regular
        # mask forms while reusing the same form-name table.
        if mon.get('name') == 'Ogerpon':
            tera_index = form_index - len(table_values)
            if 0 <= tera_index < len(table_values):
                return False, f'{table_values[tera_index]}-Tera'

        return False, None

    # If the parsed form table already contains the live species ID, use it.
    if mon_num in table:
        keys = list(table.keys())
        return keys[0] == mon_num, table[mon_num]

    # Compatibility fallback for Ogerpon layouts where Tera form IDs are
    # offset by four from the regular mask forms.
    if mon.get('name') == 'Ogerpon' and (mon_num - 4) in table:
        return False, f'{table[mon_num - 4]}-Tera'

    return False, None


def parse_mon(struct_init: NamedInitializer,
              ability_names: list[str],
              item_names: list[str],
              form_tables: dict[str, dict[int, str]],
              form_table_species_ids: dict[str, list[int]],
              level_up_learnsets: dict[str, dict[str, list[int]]],
              teachable_learnsets: dict[str, dict[str, list[str]]],
              national_dex: dict[str, int]) -> tuple[dict, list, dict, dict]:
    init_list = struct_init.expr.exprs
    mon = {}
    mon['num'] = extract_int(struct_init.name[0])
    mon['baseStats'] = {}
    mon['types'] = []
    mon['evYields'] = {}
    mon['items'] = {}
    mon['eggGroups'] = []

    evos = []
    lvlup_learnset = {}
    teach_learnset = {}

    for field_init in init_list:
        field_name = field_init.name[0].name
        field_expr = field_init.expr

        match field_name:
            case 'baseHP':
                mon['baseStats']['hp'] = extract_int(field_expr)
            case 'baseAttack':
                mon['baseStats']['atk'] = extract_int(field_expr)
            case 'baseDefense':
                mon['baseStats']['def'] = extract_int(field_expr)
            case 'baseSpeed':
                mon['baseStats']['spe'] = extract_int(field_expr)
            case 'baseSpAttack':
                mon['baseStats']['spa'] = extract_int(field_expr)
            case 'baseSpDefense':
                mon['baseStats']['spd'] = extract_int(field_expr)
            case 'types':
                types = [extract_damage_type(t) for t in field_expr.exprs]
                unique_types = []
                [unique_types.append(t) for t in types if t not in unique_types]
                mon['types'].extend(unique_types)
            case 'catchRate':
                mon['catchRate'] = extract_int(field_expr)
            case 'expYield':
                mon['expYield'] = extract_int(field_expr)
            case 'evYield_HP':
                mon['evYields']['hp'] = extract_int(field_expr)
            case 'evYield_Attack':
                mon['evYields']['atk'] = extract_int(field_expr)
            case 'evYield_Defense':
                mon['evYields']['def'] = extract_int(field_expr)
            case 'evYield_Speed':
                mon['evYields']['spe'] = extract_int(field_expr)
            case 'evYield_SpAttack':
                mon['evYields']['spa'] = extract_int(field_expr)
            case 'evYield_SpDefense':
                mon['evYields']['spd'] = extract_int(field_expr)
            case 'genderRatio':
                # If the field expression is a constant, then we can just pull it out and do a quick
                # evaluation against known constants to map to the appropriate gender
                gendered = True
                female = 0
                if isinstance(field_expr, Constant):
                    val = extract_int(field_expr)
                    match val:
                        case 0xFF:
                            gendered = False
                        case 0xFE | 0x00:
                            female = val / 0xFE
                        case _:
                            raise ValueError('unrecognized gender constant: ', val)
                else:
                    # Female % is nested as a raw constant inside a ternary; we can just pull it straight out
                    female = float(field_expr.iffalse.left.left.value) / 100

                if gendered:
                    male = 1 - female
                    mon['genderRatio'] = { 'M': male, 'F': female }
                else:
                    mon['gender'] = 'N'
            case 'eggGroups':
                group_1 = extract_int(field_expr.exprs[0])
                group_2 = extract_int(field_expr.exprs[1])
                mon['eggGroups'].append(EGG_GROUP[group_1])
                if group_2 != group_1:
                    mon['eggGroups'].append(EGG_GROUP[group_2])
            case 'abilities':
                # ability index 0 in expansion means do not output to showdown
                # all mons have at least 1 ability
                ability_0 = extract_int(field_expr.exprs[0])
                ability_1, ability_H = 0, 0
                if len(field_expr.exprs) > 1:
                    ability_1 = extract_int(field_expr.exprs[1])
                if len(field_expr.exprs) > 2:
                    ability_H = extract_int(field_expr.exprs[2])

                mon['abilities'] = { '0': ability_names[ability_0] }
                if ability_1 != 0 and ability_1 != ability_0:
                    mon['abilities']['1'] = ability_names[ability_1]
                if ability_H != 0 and ability_H != ability_0:
                    mon['abilities']['H'] = ability_names[ability_H]
            case 'bodyColor':
                mon['color'] = BODY_COLOR[extract_int(field_expr)]
            case 'speciesName':
                name = extract_u8_str(field_expr).replace('♂', '-M').replace('♀', '-F')
                if name == '??????????':
                    name = 'MissingNo.'
                mon['name'] = name
            case 'natDexNum':
                mon['nationalDex'] = national_dex[extract_id(field_expr)]
            case 'height':
                # Stored in expansion as M * 10
                mon['heightm'] = extract_int(field_expr) / 10
            case 'weight':
                # Stored in expansion as KG * 10
                mon['weightkg'] = extract_int(field_expr) / 10
            case 'itemRare':
                mon['items']['R'] = item_names[extract_int(field_expr)]
            case 'itemUncommon':
                mon['items']['U'] = item_names[extract_int(field_expr)]
            case 'formSpeciesIdTable':
                # Use the form-table membership from the CURRENT gSpeciesInfo
                # instead of assuming numeric species IDs are unchanged.
                table_name = extract_id(field_expr)
                table = form_tables.get(table_name)

                if not table:
                    # A missing/disabled form table should not stop the rest of
                    # the species data from exporting.
                    continue

                table_vals = list(table.values())
                current_ids = form_table_species_ids.get(table_name, [])

                # A one-entry table has no alternate form metadata to build.
                if len(table_vals) == 1:
                    continue

                is_base_form, resolved_form_name = resolve_dynamic_form_name(
                    mon,
                    table_name,
                    table,
                    form_table_species_ids,
                )

                if is_base_form:
                    if table_vals[0] != 'Base':
                        mon['baseForme'] = table_vals[0]

                    # Ugly Urshifu hack
                    if mon['name'] == 'Urshifu':
                        mon['formeOrder'] = [
                            mon['name'] + (f'-{table_vals[i].replace("-Style", "")}' if i > 0 else '')
                            for i in range(len(table_vals))
                        ]
                    # Ugly Xerneas hack
                    elif mon['name'] == 'Xerneas':
                        mon['formeOrder'] = ['Xerneas', 'Xerneas-Neutral']
                        mon['baseForme'] = 'Active'
                    # Ugly Vivillon hack
                    elif mon['name'] == 'Vivillon':
                        # expansion stores vivillon icy snow as the default form; showdown expects meadow to be the default
                        mon['formeOrder'] = [f'{mon["name"]}', f'{mon["name"]}-Icy-Snow']
                        mon['formeOrder'].extend([
                            f'{mon["name"]}-{table_vals[i]}'
                            for i in range(len(table_vals))
                            if table_vals[i] not in ('Base', COSMETIC_FORME_SPECIES[mon['name']].base)
                        ])
                    # Ugly Minior hack
                    elif mon['name'] == 'Minior':
                        # expansion stores minior-meteor-red as the default form; showdown indexes core-red as the default
                        mon['formeOrder'] = [f'{mon["name"]}', f'{mon["name"]}-Meteor']
                        mon['formeOrder'].extend([
                            f'{mon["name"]}-{table_vals[i]}'
                            for i in range(len(table_vals))
                            if table_vals[i] not in ('Base', COSMETIC_FORME_SPECIES[mon['name']].base)
                            and 'Meteor' not in table_vals[i]
                        ])
                    # Ugly Zygarde hack
                    elif mon['name'] == 'Zygarde':
                        mon['formeOrder'] = ['Zygarde', 'Zygarde-10%', 'Zygarde-Complete']
                        mon['baseForme'] = '50%'
                    # Ugly Greninja hack
                    elif mon['name'] == 'Greninja':
                        mon['formeOrder'] = ['Greninja', 'Greninja-Ash']
                        mon['baseForme'] = 'Base'
                    else:
                        mon['formeOrder'] = [
                            mon['name'] + (f'-{table_vals[i]}' if i > 0 else '')
                            for i in range(len(table_vals))
                        ]

                    # Cosmetic Formes
                    cosmetics = COSMETIC_FORME_SPECIES.get(mon['name'], None)
                    if cosmetics:
                        if cosmetics.alts is not None:
                            mon['cosmeticFormes'] = [
                                f'{mon["name"]}-{table_vals[i]}'
                                for i in range(len(table_vals))
                                if table_vals[i] not in ('Base', '', cosmetics.base)
                                and table_vals[i] not in cosmetics.alts
                                and (
                                    cosmetics.exclude_pattern is None
                                    or not re.match(cosmetics.exclude_pattern, table_vals[i])
                                )
                            ]
                            mon['baseForme'] = cosmetics.base

                            if cosmetics.alts:
                                mon['otherFormes'] = list(
                                    map(lambda alt: f'{mon["name"]}-{alt}', cosmetics.alts)
                                )
                    else:
                        mon['otherFormes'] = mon['formeOrder'][1:]

                else:
                    # If gSpeciesInfo says this species belongs to the form
                    # table but the form-name table is shorter, do not guess or
                    # crash. Export the species without forme metadata and show
                    # enough information to diagnose the table.
                    if resolved_form_name is None:
                        if current_ids and mon['num'] in current_ids:
                            print(
                                f'warning: could not dynamically map species '
                                f'{mon.get("name", "?")} (id {mon["num"]}) in '
                                f'{table_name}; live ids={current_ids}, '
                                f'form values={table_vals}'
                            )
                        continue

                    form_name = resolved_form_name

                    # Xerneas' alternate entry is represented specially by Showdown.
                    if mon['name'] == 'Xerneas':
                        form_name = 'Neutral'
                    # Urshifu form names contain -Style in Expansion.
                    elif mon['name'] == 'Urshifu':
                        form_name = form_name.replace('-Style', '')

                    mon['baseSpecies'] = mon['name']
                    mon['forme'] = form_name
                    mon['name'] = f'{mon["name"]}-{form_name}'
            case 'evolutions':
                # general schema from expansion: [method_id, method_param, target_species]
                for evo_method in field_expr.init.exprs:
                    method_id = extract_int(evo_method.exprs[0])
                    if method_id == 0xFFFF:
                        break

                    # explicit "no evolution" entry used by newer expansion data
                    if method_id == 0:
                        continue

                    if method_id == 0xFFFE:
                        continue

                    if method_id == ExpansionEvoMethod.SPECIFIC_MAP.value: # TODO:: Leafeon, Glaceon
                        continue

                    try:
                        method = ExpansionEvoMethod(method_id)
                    except ValueError:
                        # ignore unsupported or unknown evolution methods gracefully
                        continue

                    evos.append([method, extract_int(evo_method.exprs[1]), extract_int(evo_method.exprs[2])])
                evos.sort(key=lambda evo: evo[2])
            case 'levelUpLearnset':
                lvlup_learnset = level_up_learnsets.get(extract_id(field_expr), {})
            case 'teachableLearnset':
                teach_learnset = teachable_learnsets.get(extract_id(field_expr), {})

    return mon, evos, lvlup_learnset, teach_learnset

def zip_evos(all_data: dict,
             items: list[str],
             moves: list[str],
             map_sections: list[str]):
    for _, (mon, evos) in all_data.items():
        if not evos:
            continue

        mon['evos'] = []
        for i, evo in enumerate(evos):
            parent_dex_id = evo[2]
            parent_mon = all_data.get(parent_dex_id, (None,))[0]
            if not parent_mon:
                continue

            # Clean up Milcery's evos to Alcremie
            if mon['name'] == 'Milcery' and parent_mon['name'].startswith('Alcremie') and i > 0:
                continue

            # if the mon is already listed as evolving into this parent, continue
            # we only want to register a single evolution method per child-parent pair
            if parent_mon['name'] in mon['evos']:
                continue

            mon['evos'].append(parent_mon['name'])
            parent_mon['prevo'] = mon['name']
            method, param = evo[0], evo[1]

            match method:
                # These evo methods have no additional parameter in showdown
                case ExpansionEvoMethod.FRIENDSHIP \
                    | ExpansionEvoMethod.FRIENDSHIP_DAY \
                    | ExpansionEvoMethod.FRIENDSHIP_NIGHT \
                    | ExpansionEvoMethod.TRADE \
                    | ExpansionEvoMethod.BEAUTY \
                    | ExpansionEvoMethod.CRITICAL_HITS \
                    | ExpansionEvoMethod.SCRIPT_TRIGGER_DMG \
                    | ExpansionEvoMethod.DARK_SCROLL \
                    | ExpansionEvoMethod.WATER_SCROLL \
                    | ExpansionEvoMethod.RECOIL_DAMAGE_MALE \
                    | ExpansionEvoMethod.RECOIL_DAMAGE_FEMALE \
                    | ExpansionEvoMethod.DEFEAT_WITH_ITEM \
                    | ExpansionEvoMethod.OVERWORLD_STEPS:
                    pass

                # These evo methods interpret the parameter as a minimum level
                case ExpansionEvoMethod.LEVEL \
                    | ExpansionEvoMethod.LEVEL_ATK_GT_DEF \
                    | ExpansionEvoMethod.LEVEL_ATK_EQ_DEF \
                    | ExpansionEvoMethod.LEVEL_ATK_LT_DEF \
                    | ExpansionEvoMethod.LEVEL_SILCOON \
                    | ExpansionEvoMethod.LEVEL_CASCOON \
                    | ExpansionEvoMethod.LEVEL_SHEDINJA \
                    | ExpansionEvoMethod.LEVEL_NINJASK \
                    | ExpansionEvoMethod.LEVEL_FEMALE \
                    | ExpansionEvoMethod.LEVEL_MALE \
                    | ExpansionEvoMethod.LEVEL_NIGHT \
                    | ExpansionEvoMethod.LEVEL_DAY \
                    | ExpansionEvoMethod.LEVEL_DUSK \
                    | ExpansionEvoMethod.LEVEL_RAIN \
                    | ExpansionEvoMethod.LEVEL_DARK_TYPE_MON_IN_PARTY \
                    | ExpansionEvoMethod.LEVEL_NATURE_AMPED \
                    | ExpansionEvoMethod.LEVEL_NATURE_LOW_KEY \
                    | ExpansionEvoMethod.LEVEL_FOG \
                    | ExpansionEvoMethod.LEVEL_FAMILY_OF_THREE \
                    | ExpansionEvoMethod.LEVEL_FAMILY_OF_FOUR:
                    parent_mon['evoLevel'] = param

                # These evo methods interpret the parameter as a specific item
                case ExpansionEvoMethod.TRADE_ITEM \
                    | ExpansionEvoMethod.ITEM \
                    | ExpansionEvoMethod.ITEM_HOLD_DAY \
                    | ExpansionEvoMethod.ITEM_HOLD_NIGHT \
                    | ExpansionEvoMethod.ITEM_MALE \
                    | ExpansionEvoMethod.ITEM_FEMALE \
                    | ExpansionEvoMethod.ITEM_NIGHT \
                    | ExpansionEvoMethod.ITEM_DAY \
                    | ExpansionEvoMethod.ITEM_HOLD \
                    | ExpansionEvoMethod.ITEM_COUNT_999:
                    parent_mon['evoItem'] = items[param]

                # These evo methods interpret the parameter as a specific move
                case ExpansionEvoMethod.MOVE \
                    | ExpansionEvoMethod.MOVE_TWO_SEGMENT \
                    | ExpansionEvoMethod.MOVE_THREE_SEGMENT \
                    | ExpansionEvoMethod.USE_MOVE_TWENTY_TIMES:
                    parent_mon['evoMove'] = moves[param]
                    pass

                # These evo methods interpret the parameter as another species
                case ExpansionEvoMethod.SPECIFIC_MON_IN_PARTY \
                    | ExpansionEvoMethod.TRADE_SPECIFIC_MON:
                    parent_mon['evoSpecies'] = all_data[param][0]['name']

                # This evo method interprets the parameter as a damage type
                case ExpansionEvoMethod.FRIENDSHIP_MOVE_TYPE:
                    parent_mon['evoMove'] = f'a {DAMAGE_TYPE[param]}-type move'

                # These evo methods interpret the parameter as a specific map zone
                case ExpansionEvoMethod.MAPSEC:
                    parent_mon['evoMap'] = map_sections[param]

                case _:
                    raise ValueError('Unimplemented evo method: ', evo[0])

            descriptor = EVO_METHOD[method.value]
            parent_mon['evoType'] = descriptor.type
            parent_mon['evoCondition'] = descriptor.condition

def zip_learnsets(lvlup_learnset: dict[str, list[int]],
                  teach_learnset: dict[str, list[str]]) -> dict:
    full_learnset = defaultdict(list)
    for move, levels in lvlup_learnset.items():
        full_learnset[move] = [f'L{level}' for level in levels]
    for method, moves in teach_learnset.items():
        for move in moves:
            full_learnset[move].append(method.upper())

    return full_learnset

def parse_species_data(species_data: ExprList,
                       abilities: list[str],
                       items: list[str],
                       moves: list[str],
                       forms: dict[str, dict[int, str]],
                       map_sections: list[str],
                       level_up_learnsets: dict[str, dict[str, list[int]]],
                       teachable_learnsets: dict[str, dict[str, list[str]]],
                       national_dex: dict[str, int],
                       included_mons: list[str]) -> tuple[dict, dict]:
    # first pass: raw AST parse, build evolutions table
    all_species_data = {}
    all_learnsets = {}
    key: str

    form_table_species_ids = build_form_table_species_ids(species_data)

    for species_init in species_data:
        try:
            mon, evos, lvlup_learnset, teach_learnset = parse_mon(species_init, abilities, items, forms, form_table_species_ids, level_up_learnsets, teachable_learnsets, national_dex)
            all_species_data[mon['num']] = (mon, evos)

            if 'name' not in mon or not mon['name']:
                continue

            if mon['name'].rfind('-') != -1:
                base_name = mon['name'].split('-')[0]
                cosmetics = COSMETIC_FORME_SPECIES.get(base_name, None)
                if cosmetics and any(map(lambda s: s[0]['name'] == base_name, all_species_data.values())):
                    if cosmetics.alts is None or mon['name'] not in map(lambda alt: f'{base_name}-{alt}', cosmetics.alts):
                        mon['cosmetic'] = True # use this later during cleanup to map the name to its base form
                        continue

                special = SPECIAL_ABILITIES.get(base_name, None)
                if special:
                    form_name = mon['name'].replace(base_name, '')[1:]
                    if '-' in form_name:
                        sub_form = f'-{form_name.split("-")[0]}'
                        special_form = '-'.join(form_name.split('-')[1:])
                    else:
                        sub_form = ''
                        special_form = form_name

                    if special.form == special_form:
                        target = f'{base_name}{sub_form}'
                        parent_mon = next(
                            s for s in all_species_data.values()
                            if s[0]['name'] == target or ('baseForme' in s[0] and f'{s[0]["name"]}-{s[0]["baseForme"]}' == target)
                        )[0]
                        parent_mon['abilities']['S'] = special.ability
                        del all_species_data[mon['num']]
                        continue

            key = name_key(mon['name'])
            all_learnsets[key] = {}
            all_learnsets[key]['learnset'] = {}

            if lvlup_learnset or teach_learnset:
                all_learnsets[key]['learnset'] = zip_learnsets(lvlup_learnset, teach_learnset)
        except Exception as err:
            print('error parsing species info')
            print(species_init.show())
            raise err

    # second pass: re-target evos from source mon to target mon
    zip_evos(all_species_data, items, moves, map_sections)

    # re-zip the whole dictionary keyed according to showdown's key format
    # and flag mons which are not available
    final_species = {}
    for mon, _ in all_species_data.values():
        if 'name' not in mon or not mon['name']: # egg has no name; don't try
            continue

        if included_mons:
            mon['tier'] = 'obtainable' if mon['name'] in included_mons else 'unobtainable'

        final_species[name_key(mon['name'])] = mon

    return final_species, all_learnsets

def parse_species(fname: pathlib.Path,
                  abilities: list[str],
                  items: list[str],
                  moves: list[str],
                  forms: dict[str, dict[int, str]],
                  map_sections: list[str],
                  level_up_learnsets: dict[str, dict[str, list[int]]],
                  teachable_learnsets: dict[str, dict[str, list[str]]],
                  national_dex: dict[str, int],
                  included_mons: list[str]) -> tuple[dict, dict]:
    species_data: ExprList
    with yaspin(text=f'Loading species data: {fname}', color='cyan') as spinner:
        species_exts = load_data(fname, extra_includes=[
            r'-include', r'constants/moves.h',
        ])
        species_data = []
        for entry in reversed(species_exts):
            if not isinstance(entry, Decl):
                continue
            if entry.name == 'gSpeciesInfo' and entry.init and hasattr(entry.init, 'exprs'):
                species_data = entry.init.exprs
                break
        if not species_data:
            raise ValueError('failed to locate gSpeciesInfo in species_info.h')
        spinner.ok("✅")

    return parse_species_data(
        species_data,
        abilities,
        items,
        moves,
        forms,
        map_sections,
        level_up_learnsets,
        teachable_learnsets,
        national_dex,
        included_mons,
    )
