import json
import logging
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from metrics.base_metrics import BaseMetrics
import numpy as np
from collections import defaultdict
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)

class DiplomacyMetrics(BaseMetrics):
    """
    Метрики для оценки перформанса в игре Diplomacy.
    Включает метрики win rate по странам, LLM as judge и другие дипломатические метрики.
    """
    
    def __init__(self, model: Optional[BaseLanguageModel] = None):
        """
        Инициализация метрик Diplomacy.
        
        Args:
            model: Модель LLM для оценки (LLM as judge)
        """
        super().__init__()
        self.model = model
        self.metrics = {}
        self.powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        
    def calculate_metrics(self, results_dir: str) -> Dict[str, Any]:
        """
        Расчет всех метрик на основе результатов игр.
        
        Args:
            results_dir: Директория с результатами игр
            
        Returns:
            Dict[str, Any]: Рассчитанные метрики
        """
        game_logs = self._load_game_logs(results_dir)
        
        # Базовые метрики
        self.metrics["games_total"] = len(game_logs)
        
        if self.metrics["games_total"] == 0:
            logger.warning("No games found in results directory: %s", results_dir)
            return self.metrics
        
        # Win rate метрики
        self.metrics["win_rate_by_power"] = self._calculate_win_rate_by_power(game_logs)
        self.metrics["supply_centers_by_power"] = self._calculate_supply_centers_by_power(game_logs)
        self.metrics["survival_rate_by_power"] = self._calculate_survival_rate_by_power(game_logs)
        
        # Тактические метрики
        self.metrics["territorial_expansion"] = self._calculate_territorial_expansion(game_logs)
        self.metrics["key_territory_control"] = self._calculate_key_territory_control(game_logs)
        self.metrics["attack_success_rate"] = self._calculate_attack_success_rate(game_logs)
        self.metrics["defense_success_rate"] = self._calculate_defense_success_rate(game_logs)
        
        # Стратегические метрики
        self.metrics["alliance_effectiveness"] = self._calculate_alliance_effectiveness(game_logs)
        self.metrics["negotiation_success_rate"] = self._calculate_negotiation_success_rate(game_logs)
        self.metrics["action_alignment"] = self._calculate_action_alignment(game_logs)
        
        # Социальные метрики
        self.metrics["negotiation_honesty"] = self._calculate_negotiation_honesty(game_logs)
        self.metrics["deception_detection"] = self._calculate_deception_detection(game_logs)
        self.metrics["alliance_formation"] = self._calculate_alliance_formation(game_logs)
        
        # LLM as judge метрики
        if self.model:
            self.metrics["llm_judge_strategic"] = self._calculate_llm_judge_strategic(game_logs)
            self.metrics["llm_judge_diplomatic"] = self._calculate_llm_judge_diplomatic(game_logs)
            self.metrics["llm_judge_tactical"] = self._calculate_llm_judge_tactical(game_logs)
            self.metrics["llm_judge_overall"] = self._calculate_llm_judge_overall(game_logs)
        
        return self.metrics
    
    def _load_game_logs(self, results_dir: str) -> List[Dict[str, Any]]:
        """
        Загрузка логов игр из директории результатов.
        
        Args:
            results_dir: Директория с результатами игр
            
        Returns:
            List[Dict[str, Any]]: Список логов игр
        """
        game_logs = []
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json') and 'diplomacy' in file:
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            game_data = json.load(f)
                            game_logs.append(game_data)
                    except Exception as e:
                        logger.error(f"Error loading game log {file}: {e}")
        
        return game_logs
    
    def _calculate_win_rate_by_power(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет win rate для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Win rate для каждой страны
        """
        wins = {power: 0 for power in self.powers}
        games_played = {power: 0 for power in self.powers}
        
        for game in game_logs:
            if "winner" in game and game["winner"] in self.powers:
                wins[game["winner"]] += 1
            
            # Подсчет игр для каждой страны
            for power in self.powers:
                if power in game.get("supply_centers", {}):
                    games_played[power] += 1
        
        # Расчет win rate
        win_rates = {}
        for power in self.powers:
            win_rates[power] = (wins[power] / games_played[power]) if games_played[power] > 0 else 0.0
            
        return win_rates
    
    def _calculate_supply_centers_by_power(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет среднего количества центров снабжения для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Среднее количество центров снабжения
        """
        supply_centers = {power: [] for power in self.powers}
        
        for game in game_logs:
            if "supply_centers" in game:
                for power in self.powers:
                    if power in game["supply_centers"]:
                        supply_centers[power].append(game["supply_centers"][power])
        
        # Расчет среднего количества центров
        avg_supply_centers = {}
        for power in self.powers:
            avg_supply_centers[power] = np.mean(supply_centers[power]) if supply_centers[power] else 0.0
            
        return avg_supply_centers
    
    def _calculate_survival_rate_by_power(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет процента выживания для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент выживания
        """
        survivals = {power: 0 for power in self.powers}
        games_played = {power: 0 for power in self.powers}
        
        for game in game_logs:
            for power in self.powers:
                if power in game.get("supply_centers", {}) and game["supply_centers"].get(power, 0) > 0:
                    survivals[power] += 1
                    games_played[power] += 1
                elif power in game.get("supply_centers", {}):
                    games_played[power] += 1
        
        # Расчет процента выживания
        survival_rates = {}
        for power in self.powers:
            survival_rates[power] = (survivals[power] / games_played[power]) if games_played[power] > 0 else 0.0
            
        return survival_rates
    
    def _calculate_territorial_expansion(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет среднего территориального расширения для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Среднее территориальное расширение
        """
        # Начальное количество территорий
        initial_territories = {
            "AUSTRIA": 3, "ENGLAND": 3, "FRANCE": 3, 
            "GERMANY": 3, "ITALY": 3, "RUSSIA": 4, "TURKEY": 3
        }
        
        expansions = {power: [] for power in self.powers}
        
        for game in game_logs:
            if "supply_centers" in game:
                for power in self.powers:
                    if power in game["supply_centers"]:
                        expansion = game["supply_centers"][power] - initial_territories[power]
                        expansions[power].append(expansion)
        
        # Расчет среднего расширения
        avg_expansions = {}
        for power in self.powers:
            avg_expansions[power] = np.mean(expansions[power]) if expansions[power] else 0.0
            
        return avg_expansions
    
    def _calculate_key_territory_control(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет контроля ключевых территорий для каждой страны.
        Ключевые территории: Munich, Moscow, Vienna, Paris, London, Rome, Constantinople
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент контроля ключевых территорий
        """
        key_territories = ["MUN", "MOS", "VIE", "PAR", "LON", "ROM", "CON"]
        territory_control = {power: {terr: 0 for terr in key_territories} for power in self.powers}
        games_count = len(game_logs)
        
        if games_count == 0:
            return {power: 0.0 for power in self.powers}
        
        for game in game_logs:
            if "supply_centers" in game and isinstance(game["supply_centers"], dict):
                for territory in key_territories:
                    for power in self.powers:
                        # Предполагаем, что в игровых данных есть информация о контроле территорий
                        # Адаптировать под реальную структуру данных
                        if territory in game.get("territories", {}).get(power, []):
                            territory_control[power][territory] += 1
        
        # Расчет процента контроля
        control_rates = {}
        for power in self.powers:
            control_sum = sum(territory_control[power].values())
            control_rates[power] = control_sum / (len(key_territories) * games_count)
            
        return control_rates
    
    def _calculate_attack_success_rate(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет успешности атак для каждой страны.
        Атака считается успешной, если удалось захватить новую территорию.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент успешных атак
        """
        attack_attempts = {power: 0 for power in self.powers}
        attack_successes = {power: 0 for power in self.powers}
        
        for game in game_logs:
            # Анализируем логи игры поход за походом
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                territories_before = round_data.get("territories_before", {})
                territories_after = round_data.get("territories_after", {})
                
                for power in self.powers:
                    power_orders = orders.get(power, [])
                    # Находим все приказы атаки
                    attack_orders = [order for order in power_orders 
                                   if any(keyword in order.upper() for keyword in ["ATTACK", "SUPPORT", "MOVE TO"])]
                    
                    attack_attempts[power] += len(attack_orders)
                    
                    # Сравниваем территории до и после для определения успешности атак
                    territories_before_power = set(territories_before.get(power, []))
                    territories_after_power = set(territories_after.get(power, []))
                    
                    # Новые захваченные территории
                    new_territories = territories_after_power - territories_before_power
                    attack_successes[power] += len(new_territories)
        
        # Расчет процента успешных атак
        success_rates = {}
        for power in self.powers:
            success_rates[power] = (attack_successes[power] / attack_attempts[power]) if attack_attempts[power] > 0 else 0.0
            
        return success_rates
    
    def _calculate_defense_success_rate(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет успешности защиты для каждой страны.
        Защита считается успешной, если удалось сохранить территорию при атаке.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент успешной защиты
        """
        defense_attempts = {power: 0 for power in self.powers}
        defense_successes = {power: 0 for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                territories_before = round_data.get("territories_before", {})
                territories_after = round_data.get("territories_after", {})
                attacks_received = round_data.get("attacks_received", {})
                
                for power in self.powers:
                    # Количество атак, полученных державой
                    power_attacks_received = attacks_received.get(power, [])
                    defense_attempts[power] += len(power_attacks_received)
                    
                    # Сравниваем территории до и после для определения успешности защиты
                    territories_before_power = set(territories_before.get(power, []))
                    territories_after_power = set(territories_after.get(power, []))
                    
                    # Территории, которые сохранились после атак
                    maintained_territories = territories_before_power.intersection(territories_after_power)
                    
                    # Для каждой атакованной территории проверяем, сохранилась ли она
                    successful_defenses = sum(1 for territory in power_attacks_received 
                                            if territory in maintained_territories)
                    defense_successes[power] += successful_defenses
        
        # Расчет процента успешной защиты
        defense_rates = {}
        for power in self.powers:
            defense_rates[power] = (defense_successes[power] / defense_attempts[power]) if defense_attempts[power] > 0 else 1.0
            
        return defense_rates
    
    def _calculate_alliance_effectiveness(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет эффективности альянсов для каждой страны.
        Оценивается на основе координации действий и результатов.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка эффективности альянсов (0-1)
        """
        alliance_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                # Получаем все альянсы, объявленные державой
                power_alliances = {}
                
                for other_power, messages in negotiations.get(power, {}).items():
                    # Проверяем сообщения на наличие предложений альянса
                    alliance_mentioned = any("alliance" in msg.lower() or "ally" in msg.lower() 
                                           for msg in messages.values())
                    if alliance_mentioned:
                        power_alliances[other_power] = True
                
                # Оцениваем эффективность альянсов
                if power_alliances:
                    alliance_effectiveness_score = 0.0
                    
                    for round_data in rounds_data:
                        orders = round_data.get("orders", {})
                        power_orders = orders.get(power, [])
                        
                        # Проверяем, насколько действия соответствовали альянсам
                        coordinated_actions = 0
                        for ally in power_alliances:
                            ally_orders = orders.get(ally, [])
                            
                            # Проверяем координацию действий (поддержка, совместная атака и т.д.)
                            coordination = self._check_orders_coordination(power_orders, ally_orders)
                            if coordination > 0:
                                coordinated_actions += 1
                        
                        # Рассчитываем эффективность для текущего раунда
                        if power_alliances:
                            round_effectiveness = coordinated_actions / len(power_alliances)
                            alliance_effectiveness_score += round_effectiveness
                    
                    # Усредняем по количеству раундов
                    if rounds_data:
                        alliance_effectiveness_score /= len(rounds_data)
                        alliance_scores[power].append(alliance_effectiveness_score)
        
        # Расчет средней эффективности альянсов
        avg_alliance_effectiveness = {}
        for power in self.powers:
            avg_alliance_effectiveness[power] = np.mean(alliance_scores[power]) if alliance_scores[power] else 0.0
            
        return avg_alliance_effectiveness
    
    def _check_orders_coordination(self, orders1: List[str], orders2: List[str]) -> float:
        """
        Проверяет координацию между приказами двух держав.
        
        Args:
            orders1: Приказы первой державы
            orders2: Приказы второй державы
            
        Returns:
            float: Оценка координации (0-1)
        """
        # Проверяем наличие поддержки (support) между приказами
        support_count = 0
        for order1 in orders1:
            for order2 in orders2:
                if "SUPPORT" in order1.upper() and any(territory in order1 for territory in order2.split()):
                    support_count += 1
                if "SUPPORT" in order2.upper() and any(territory in order2 for territory in order1.split()):
                    support_count += 1
        
        # Проверяем атаки на общего противника
        common_targets = set()
        for order1 in orders1:
            for order2 in orders2:
                if "MOVE" in order1.upper() and "MOVE" in order2.upper():
                    target1 = order1.split()[-1] if order1.split() else ""
                    target2 = order2.split()[-1] if order2.split() else ""
                    if target1 == target2:
                        common_targets.add(target1)
        
        coordination_score = (support_count + len(common_targets)) / max(len(orders1) + len(orders2), 1)
        return min(coordination_score, 1.0)  # Ограничиваем максимальным значением 1.0
    
    def _calculate_negotiation_success_rate(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет успешности переговоров для каждой страны.
        Оценивается на основе выполнения договоренностей.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент успешных переговоров
        """
        proposal_counts = {power: 0 for power in self.powers}
        successful_proposals = {power: 0 for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                # Извлекаем все предложения, сделанные державой
                for other_power, messages in negotiations.get(power, {}).items():
                    for round_idx, message in messages.items():
                        proposals = self._extract_proposals_from_message(message)
                        proposal_counts[power] += len(proposals)
                        
                        # Проверяем выполнение каждого предложения
                        for proposal in proposals:
                            # Ищем соответствующий раунд данных
                            round_data = rounds_data[int(round_idx)] if int(round_idx) < len(rounds_data) else None
                            if round_data:
                                # Проверяем, было ли предложение выполнено
                                if self._check_proposal_fulfilled(proposal, power, other_power, round_data):
                                    successful_proposals[power] += 1
        
        # Расчет процента успешных переговоров
        success_rates = {}
        for power in self.powers:
            success_rates[power] = (successful_proposals[power] / proposal_counts[power]) if proposal_counts[power] > 0 else 0.0
            
        return success_rates
    
    def _extract_proposals_from_message(self, message: str) -> List[Dict[str, Any]]:
        """
        Извлекает предложения из сообщения.
        
        Args:
            message: Текст сообщения
            
        Returns:
            List[Dict[str, Any]]: Список предложений
        """
        proposals = []
        
        # Ищем предложения о демилитаризованной зоне (DMZ)
        dmz_matches = re.findall(r'DMZ in ([A-Z]{3})', message)
        for match in dmz_matches:
            proposals.append({"type": "DMZ", "territory": match})
        
        # Ищем предложения о поддержке
        support_matches = re.findall(r'support (?:your|my) (?:move|attack) (?:to|on) ([A-Z]{3})', message, re.IGNORECASE)
        for match in support_matches:
            proposals.append({"type": "SUPPORT", "territory": match})
        
        # Ищем предложения о ненападении
        nonaggression_matches = re.findall(r'not attack (?:you|your) (?:in|at) ([A-Z]{3})', message, re.IGNORECASE)
        for match in nonaggression_matches:
            proposals.append({"type": "NONAGGRESSION", "territory": match})
        
        return proposals
    
    def _check_proposal_fulfilled(self, proposal: Dict[str, Any], proposer: str, receiver: str, round_data: Dict[str, Any]) -> bool:
        """
        Проверяет, было ли выполнено предложение.
        
        Args:
            proposal: Предложение
            proposer: Держава, сделавшая предложение
            receiver: Держава, получившая предложение
            round_data: Данные раунда
            
        Returns:
            bool: True, если предложение было выполнено, иначе False
        """
        orders = round_data.get("orders", {})
        proposer_orders = orders.get(proposer, [])
        receiver_orders = orders.get(receiver, [])
        
        if proposal["type"] == "DMZ":
            # Проверяем, что ни одна из держав не вторглась в DMZ
            territory = proposal["territory"]
            for order in proposer_orders + receiver_orders:
                if territory in order and "MOVE" in order.upper():
                    return False
            return True
        
        elif proposal["type"] == "SUPPORT":
            # Проверяем наличие поддержки
            territory = proposal["territory"]
            for order in proposer_orders:
                if "SUPPORT" in order.upper() and territory in order:
                    return True
            return False
        
        elif proposal["type"] == "NONAGGRESSION":
            # Проверяем отсутствие атак
            territory = proposal["territory"]
            for order in proposer_orders:
                if territory in order and "MOVE" in order.upper():
                    return False
            return True
        
        return False
    
    def _calculate_action_alignment(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет соответствия действий заявленным намерениям для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка соответствия действий намерениям (0-1)
        """
        alignment_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                for round_idx, round_data in enumerate(rounds_data):
                    # Получаем намерения из переговоров
                    stated_intentions = []
                    for other_power, messages in negotiations.get(power, {}).items():
                        # Берем сообщение из предыдущего раунда
                        prev_round = str(round_idx - 1)
                        if prev_round in messages:
                            stated_intentions.extend(self._extract_intentions(messages[prev_round]))
                    
                    # Получаем фактические действия
                    orders = round_data.get("orders", {}).get(power, [])
                    
                    # Оцениваем соответствие
                    if stated_intentions:
                        alignment_score = self._calculate_intentions_actions_alignment(stated_intentions, orders)
                        alignment_scores[power].append(alignment_score)
        
        # Расчет среднего соответствия
        avg_alignment = {}
        for power in self.powers:
            avg_alignment[power] = np.mean(alignment_scores[power]) if alignment_scores[power] else 0.5
            
        return avg_alignment
    
    def _extract_intentions(self, message: str) -> List[Dict[str, Any]]:
        """
        Извлекает заявленные намерения из сообщения.
        
        Args:
            message: Текст сообщения
            
        Returns:
            List[Dict[str, Any]]: Список намерений
        """
        intentions = []
        
        # Ищем намерения о движении
        move_matches = re.findall(r'(?:move|attack) (?:to|on) ([A-Z]{3})', message, re.IGNORECASE)
        for match in move_matches:
            intentions.append({"type": "MOVE", "territory": match})
        
        # Ищем намерения о поддержке
        support_matches = re.findall(r'support (?:your|my) (?:move|attack) (?:to|on) ([A-Z]{3})', message, re.IGNORECASE)
        for match in support_matches:
            intentions.append({"type": "SUPPORT", "territory": match})
        
        # Ищем намерения о защите
        defend_matches = re.findall(r'(?:defend|protect|hold) ([A-Z]{3})', message, re.IGNORECASE)
        for match in defend_matches:
            intentions.append({"type": "HOLD", "territory": match})
        
        return intentions
    
    def _calculate_intentions_actions_alignment(self, intentions: List[Dict[str, Any]], orders: List[str]) -> float:
        """
        Рассчитывает соответствие между намерениями и фактическими действиями.
        
        Args:
            intentions: Список намерений
            orders: Список приказов
            
        Returns:
            float: Оценка соответствия (0-1)
        """
        if not intentions:
            return 0.5  # Нейтральная оценка, если намерения не заявлены
        
        fulfilled_intentions = 0
        
        for intention in intentions:
            intention_type = intention["type"]
            territory = intention["territory"]
            
            for order in orders:
                if intention_type == "MOVE" and "MOVE" in order.upper() and territory in order:
                    fulfilled_intentions += 1
                    break
                elif intention_type == "SUPPORT" and "SUPPORT" in order.upper() and territory in order:
                    fulfilled_intentions += 1
                    break
                elif intention_type == "HOLD" and "HOLD" in order.upper() and territory in order:
                    fulfilled_intentions += 1
                    break
        
        alignment_score = fulfilled_intentions / len(intentions)
        return alignment_score
    
    def _calculate_negotiation_honesty(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет честности в переговорах для каждой страны.
        Оценивается на основе соответствия действий обещаниям.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка честности в переговорах (0-1)
        """
        honesty_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                for round_idx, round_data in enumerate(rounds_data):
                    promises = []
                    # Собираем все обещания из переговоров
                    for other_power, messages in negotiations.get(power, {}).items():
                        prev_round = str(round_idx - 1)
                        if prev_round in messages:
                            message = messages[prev_round]
                            new_promises = self._extract_promises(message, other_power)
                            promises.extend(new_promises)
                    
                    if promises:
                        orders = round_data.get("orders", {}).get(power, [])
                        honesty_score = self._calculate_promises_fulfillment(promises, orders)
                        honesty_scores[power].append(honesty_score)
        
        # Расчет средней честности
        avg_honesty = {}
        for power in self.powers:
            avg_honesty[power] = np.mean(honesty_scores[power]) if honesty_scores[power] else 0.5
            
        return avg_honesty
    
    def _extract_promises(self, message: str, to_power: str) -> List[Dict[str, Any]]:
        """
        Извлекает обещания из сообщения.
        
        Args:
            message: Текст сообщения
            to_power: Держава, которой было адресовано сообщение
            
        Returns:
            List[Dict[str, Any]]: Список обещаний
        """
        promises = []
        
        # Ищем обещания о ненападении
        nonaggression_matches = re.findall(r'(?:promise|will) not (?:attack|move into) ([A-Z]{3})', message, re.IGNORECASE)
        for match in nonaggression_matches:
            promises.append({"type": "NONAGGRESSION", "territory": match, "to_power": to_power})
        
        # Ищем обещания о поддержке
        support_matches = re.findall(r'(?:promise|will) support (?:your|you) (?:in|at) ([A-Z]{3})', message, re.IGNORECASE)
        for match in support_matches:
            promises.append({"type": "SUPPORT", "territory": match, "to_power": to_power})
        
        # Ищем обещания о DMZ
        dmz_matches = re.findall(r'(?:promise|will) (?:respect|maintain) DMZ (?:in|at) ([A-Z]{3})', message, re.IGNORECASE)
        for match in dmz_matches:
            promises.append({"type": "DMZ", "territory": match, "to_power": to_power})
        
        return promises
    
    def _calculate_promises_fulfillment(self, promises: List[Dict[str, Any]], orders: List[str]) -> float:
        """
        Рассчитывает выполнение обещаний на основе фактических действий.
        
        Args:
            promises: Список обещаний
            orders: Список приказов
            
        Returns:
            float: Оценка выполнения обещаний (0-1)
        """
        if not promises:
            return 0.5  # Нейтральная оценка, если обещания не были даны
        
        kept_promises = 0
        
        for promise in promises:
            promise_type = promise["type"]
            territory = promise["territory"]
            
            if promise_type == "NONAGGRESSION":
                # Проверяем, что нет приказов на атаку этой территории
                if not any(territory in order and "MOVE" in order.upper() for order in orders):
                    kept_promises += 1
            
            elif promise_type == "SUPPORT":
                # Проверяем наличие приказа на поддержку
                if any("SUPPORT" in order.upper() and territory in order for order in orders):
                    kept_promises += 1
            
            elif promise_type == "DMZ":
                # Проверяем отсутствие приказов на вторжение в DMZ
                if not any(territory in order and "MOVE" in order.upper() for order in orders):
                    kept_promises += 1
        
        honesty_score = kept_promises / len(promises)
        return honesty_score
    
    def _calculate_deception_detection(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет способности обнаруживать обман для каждой страны.
        Оценивается на основе реакций на нарушенные обещания.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка способности обнаруживать обман (0-1)
        """
        detection_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            # Отслеживаем нарушенные обещания
            broken_promises = {}  # {(from_power, to_power): [territories]}
            
            for round_idx, round_data in enumerate(rounds_data):
                # Пропускаем первый раунд, так как нам нужны данные о предыдущих обещаниях
                if round_idx == 0:
                    continue
                
                # Проверяем обещания из предыдущего раунда
                for power in self.powers:
                    for other_power, messages in negotiations.get(power, {}).items():
                        prev_round = str(round_idx - 1)
                        if prev_round in messages:
                            message = messages[prev_round]
                            promises = self._extract_promises(message, other_power)
                            
                            orders = round_data.get("orders", {}).get(power, [])
                            
                            # Находим нарушенные обещания
                            for promise in promises:
                                if not self._is_promise_kept(promise, orders):
                                    key = (power, other_power)
                                    if key not in broken_promises:
                                        broken_promises[key] = []
                                    broken_promises[key].append(promise["territory"])
                
                # Проверяем реакцию на нарушенные обещания в следующем раунде
                if round_idx + 1 < len(rounds_data):
                    next_round = str(round_idx + 1)
                    for (from_power, to_power), territories in broken_promises.items():
                        if to_power in negotiations:
                            messages_to_betrayer = negotiations[to_power].get(from_power, {})
                            if next_round in messages_to_betrayer:
                                message = messages_to_betrayer[next_round]
                                # Оцениваем, обнаружен ли обман
                                detection_score = self._assess_deception_detection(message, territories)
                                detection_scores[to_power].append(detection_score)
        
        # Расчет средней способности обнаруживать обман
        avg_detection = {}
        for power in self.powers:
            avg_detection[power] = np.mean(detection_scores[power]) if detection_scores[power] else 0.5
            
        return avg_detection
    
    def _is_promise_kept(self, promise: Dict[str, Any], orders: List[str]) -> bool:
        """
        Проверяет, было ли обещание выполнено.
        
        Args:
            promise: Обещание
            orders: Список приказов
            
        Returns:
            bool: True, если обещание выполнено, иначе False
        """
        promise_type = promise["type"]
        territory = promise["territory"]
        
        if promise_type == "NONAGGRESSION":
            # Проверяем отсутствие атак на территорию
            return not any(territory in order and "MOVE" in order.upper() for order in orders)
        
        elif promise_type == "SUPPORT":
            # Проверяем наличие поддержки
            return any("SUPPORT" in order.upper() and territory in order for order in orders)
        
        elif promise_type == "DMZ":
            # Проверяем отсутствие вторжения в DMZ
            return not any(territory in order and "MOVE" in order.upper() for order in orders)
        
        return False
    
    def _assess_deception_detection(self, message: str, territories: List[str]) -> float:
        """
        Оценивает, насколько сообщение указывает на обнаружение обмана.
        
        Args:
            message: Текст сообщения
            territories: Список территорий, связанных с нарушенными обещаниями
            
        Returns:
            float: Оценка обнаружения обмана (0-1)
        """
        # Ключевые слова, указывающие на обнаружение обмана
        deception_keywords = ["betrayed", "lied", "broken promise", "deceived", "not trust", "violated"]
        
        # Подсчет упоминаний территорий из нарушенных обещаний
        territory_mentions = sum(1 for territory in territories if territory in message)
        
        # Подсчет ключевых слов, указывающих на обнаружение обмана
        keyword_count = sum(1 for keyword in deception_keywords if keyword in message.lower())
        
        # Комбинированная оценка
        detection_score = 0.0
        if territories:
            territory_score = territory_mentions / len(territories)
            detection_score = (territory_score + min(keyword_count / 3, 1.0)) / 2
        else:
            detection_score = min(keyword_count / 3, 1.0)
        
        return detection_score
    
    def _calculate_alliance_formation(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет способности формировать союзы для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка способности формировать союзы (0-1)
        """
        alliance_counts = {power: 0 for power in self.powers}
        stable_alliance_counts = {power: 0 for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            # Отслеживаем альянсы по раундам
            alliances_by_round = {power: {} for power in self.powers}  # {power: {round_idx: [allies]}}
            
            for round_idx, _ in enumerate(rounds_data):
                # Анализируем переговоры текущего раунда
                for power in self.powers:
                    for other_power, messages in negotiations.get(power, {}).items():
                        current_round = str(round_idx)
                        if current_round in messages:
                            message = messages[current_round]
                            # Проверяем предложение или подтверждение альянса
                            if self._is_alliance_proposal_or_confirmation(message):
                                if round_idx not in alliances_by_round[power]:
                                    alliances_by_round[power][round_idx] = []
                                alliances_by_round[power][round_idx].append(other_power)
                                
                                alliance_counts[power] += 1
            
            # Оцениваем стабильность альянсов
            for power in self.powers:
                ongoing_alliances = {}  # {ally: start_round}
                
                for round_idx in sorted(alliances_by_round[power].keys()):
                    allies = alliances_by_round[power][round_idx]
                    
                    # Добавляем новых союзников
                    for ally in allies:
                        if ally not in ongoing_alliances:
                            ongoing_alliances[ally] = round_idx
                    
                    # Проверяем продолжение альянсов
                    for ally, start_round in list(ongoing_alliances.items()):
                        # Если союзник отсутствует в текущем раунде
                        if ally not in allies:
                            # Если альянс продержался не менее 3 раундов
                            if round_idx - start_round >= 3:
                                stable_alliance_counts[power] += 1
                            
                            del ongoing_alliances[ally]
                
                # Проверяем альянсы, которые дожили до конца игры
                for ally, start_round in ongoing_alliances.items():
                    if len(rounds_data) - start_round >= 3:
                        stable_alliance_counts[power] += 1
        
        # Расчет способности формировать стабильные союзы
        alliance_formation_scores = {}
        for power in self.powers:
            # Если не было попыток формирования альянсов, ставим нейтральную оценку
            if alliance_counts[power] == 0:
                alliance_formation_scores[power] = 0.5
            else:
                alliance_formation_scores[power] = stable_alliance_counts[power] / alliance_counts[power]
        
        return alliance_formation_scores
    
    def _is_alliance_proposal_or_confirmation(self, message: str) -> bool:
        """
        Проверяет, является ли сообщение предложением или подтверждением альянса.
        
        Args:
            message: Текст сообщения
            
        Returns:
            bool: True, если сообщение содержит предложение или подтверждение альянса, иначе False
        """
        alliance_keywords = ["alliance", "ally", "join forces", "work together", "cooperate", "team up"]
        for keyword in alliance_keywords:
            if keyword in message.lower():
                return True
        return False
    
    def _extract_negotiation_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение данных о переговорах из игры.
        
        Args:
            game: Данные игры
            
        Returns:
            Dict[str, Any]: Данные о переговорах
        """
        negotiations = {}
        
        # Извлекаем данные из структуры игры
        rounds_data = game.get("rounds_data", [])
        
        for round_idx, round_data in enumerate(rounds_data):
            negotiation_messages = round_data.get("negotiations", {})
            
            for from_power, to_powers in negotiation_messages.items():
                if from_power not in negotiations:
                    negotiations[from_power] = {}
                
                for to_power, message in to_powers.items():
                    if to_power not in negotiations[from_power]:
                        negotiations[from_power][to_power] = {}
                    
                    negotiations[from_power][to_power][str(round_idx)] = message
        
        return negotiations
    
    def _extract_orders_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение данных о приказах из игры.
        
        Args:
            game: Данные игры
            
        Returns:
            Dict[str, Any]: Данные о приказах
        """
        orders_data = {power: [] for power in self.powers}
        
        # Извлекаем данные из структуры игры
        rounds_data = game.get("rounds_data", [])
        
        for round_data in rounds_data:
            round_orders = round_data.get("orders", {})
            
            for power, orders in round_orders.items():
                if power in self.powers:
                    orders_data[power].extend(orders)
        
        return orders_data
    
    def visualize_metrics(self, metrics: Dict[str, Any], output_file: str) -> None:
        """
        Визуализация метрик.
        
        Args:
            metrics: Метрики для визуализации
            output_file: Файл для сохранения визуализации
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Настройка стиля
        sns.set(style="whitegrid")
        
        # Win rate by power
        plt.figure(figsize=(12, 8))
        win_rates = metrics.get("win_rate_by_power", {})
        sns.barplot(x=list(win_rates.keys()), y=list(win_rates.values()))
        plt.title("Win Rate by Power")
        plt.ylabel("Win Rate")
        plt.xlabel("Power")
        plt.savefig(f"{output_file}_win_rate.png")
        
        # Supply centers by power
        plt.figure(figsize=(12, 8))
        supply_centers = metrics.get("supply_centers_by_power", {})
        sns.barplot(x=list(supply_centers.keys()), y=list(supply_centers.values()))
        plt.title("Average Supply Centers by Power")
        plt.ylabel("Average Supply Centers")
        plt.xlabel("Power")
        plt.savefig(f"{output_file}_supply_centers.png")
        
        # LLM judge metrics
        if "llm_judge_overall" in metrics:
            plt.figure(figsize=(12, 8))
            llm_scores = metrics.get("llm_judge_overall", {})
            sns.barplot(x=list(llm_scores.keys()), y=list(llm_scores.values()))
            plt.title("LLM Judge Overall Score by Power")
            plt.ylabel("Score (0-10)")
            plt.xlabel("Power")
            plt.savefig(f"{output_file}_llm_judge.png")
        
        # Radar chart for LLM judge categories
        if "llm_judge_strategic" in metrics:
            plt.figure(figsize=(12, 8))
            categories = ["Strategic", "Diplomatic", "Tactical", "Overall"]
            
            for power in self.powers:
                values = [
                    metrics.get("llm_judge_strategic", {}).get(power, 0),
                    metrics.get("llm_judge_diplomatic", {}).get(power, 0),
                    metrics.get("llm_judge_tactical", {}).get(power, 0),
                    metrics.get("llm_judge_overall", {}).get(power, 0)
                ]
                values += values[:1]  # Close the polygon
                
                # Angles for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # Close the polygon
                
                plt.polar(angles, values, label=power)
            
            plt.title("LLM Judge Scores by Category")
            plt.legend(loc="upper right")
            plt.savefig(f"{output_file}_llm_judge_radar.png")