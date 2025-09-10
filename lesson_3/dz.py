from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum
import random


class TransportType(Enum):
    TRUCK = "Грузовик"
    VAN = "Фургон"
    REFRIGERATOR = "Рефрижератор"
    AIR = "Авиаперевозка"


class CargoType(Enum):
    GENERAL = "Общий груз"
    REFRIGERATED = "Охлаждаемый"
    DANGEROUS = "Опасный груз"
    FRAGILE = "Хрупкий"


class Warehouse:
    def __init__(self, id: int, name: str, address: str):
        self.id = id
        self.name = name
        self.address = address

    def __str__(self):
        return f"{self.name} ({self.address})"


class Cargo:
    def __init__(self, id: int, name: str, weight: float, volume: float,
                 cargo_type: CargoType, fragile: bool = False, dangerous: bool = False):
        self.id = id
        self.name = name
        self.weight = weight
        self.volume = volume
        self.cargo_type = cargo_type
        self.fragile = fragile
        self.dangerous = dangerous
        self.requires_refrigeration = cargo_type == CargoType.REFRIGERATED
        self.is_fragile = fragile
        self.is_dangerous = dangerous

    def __str__(self):
        return f"{self.name} ({self.weight}кг, {self.volume}м³)"


class Transport(ABC):
    def __init__(self, id: int, name: str, max_weight: float, max_volume: float,
                 cost_per_km: float, speed: float, transport_type: TransportType):
        self.id = id
        self.name = name
        self.max_weight = max_weight
        self.max_volume = max_volume
        self.cost_per_km = cost_per_km
        self.speed = speed
        self.transport_type = transport_type
        self.current_location: Optional[Warehouse] = None

    @abstractmethod
    def can_transport(self, cargo: Cargo) -> bool:
        pass

    @abstractmethod
    def calculate_cost_factor(self, cargo: Cargo) -> float:
        pass

    def __str__(self):
        return f"{self.name} (макс: {self.max_weight}кг, {self.max_volume}м³)"


class Truck(Transport):
    def __init__(self, id: int, name: str, max_weight: float, max_volume: float,
                 cost_per_km: float = 15.0, speed: float = 90.0):
        super().__init__(id, name, max_weight, max_volume, cost_per_km,
                         speed, TransportType.TRUCK)

    def can_transport(self, cargo: Cargo) -> bool:
        return (cargo.weight <= self.max_weight and
                cargo.volume <= self.max_volume and
                not cargo.requires_refrigeration)

    def calculate_cost_factor(self, cargo: Cargo) -> float:
        factor = 1.0
        if cargo.dangerous:
            factor *= 1.3
        if cargo.fragile:
            factor *= 1.2
        return factor


class RefrigeratorTruck(Transport):
    def __init__(self, id: int, name: str, max_weight: float, max_volume: float,
                 min_temp: float, cost_per_km: float = 20.0, speed: float = 90.0):
        super().__init__(id, name, max_weight, max_volume, cost_per_km,
                         speed, TransportType.REFRIGERATOR)
        self.min_temp = min_temp

    def can_transport(self, cargo: Cargo) -> bool:
        return (cargo.weight <= self.max_weight and
                cargo.volume <= self.max_volume)

    def calculate_cost_factor(self, cargo: Cargo) -> float:
        factor = 1.5
        if cargo.dangerous:
            factor *= 1.4
        if cargo.fragile:
            factor *= 1.3
        return factor


class AirTransport(Transport):
    def __init__(self, id: int, name: str, max_weight: float, max_volume: float,
                 cost_per_km: float = 50.0, speed: float = 800.0):
        super().__init__(id, name, max_weight, max_volume, cost_per_km,
                         speed, TransportType.AIR)

    def can_transport(self, cargo: Cargo) -> bool:
        return (cargo.weight <= self.max_weight and
                cargo.volume <= self.max_volume and
                not cargo.dangerous)

    def calculate_cost_factor(self, cargo: Cargo) -> float:
        factor = 3.0
        if cargo.fragile:
            factor *= 1.4
        if cargo.requires_refrigeration:
            factor *= 1.8
        return factor


class DistanceCalculator:
    @staticmethod
    def calculate_distance(warehouse1: Warehouse, warehouse2: Warehouse) -> float:
        addresses = [warehouse1.address.lower(), warehouse2.address.lower()]
        return random.uniform(50, 500)


class CostCalculator:
    def __init__(self, base_multiplier: float = 1.0):
        self.base_multiplier = base_multiplier

    def calculate_cost(self, transport: Transport, cargo: Cargo, distance: float) -> float:
        if not transport.can_transport(cargo):
            raise ValueError("Транспорт не может перевозить данный груз")

        base_cost = distance * transport.cost_per_km
        cargo_factor = transport.calculate_cost_factor(cargo)

        total_cost = base_cost * cargo_factor * self.base_multiplier
        return round(total_cost, 2)

    def calculate_time(self, distance: float, speed: float) -> timedelta:
        travel_hours = distance / speed
        total_hours = travel_hours + 1.0
        return timedelta(hours=total_hours)


class Delivery:
    def __init__(self, id: int, cargo: Cargo, origin: Warehouse, destination: Warehouse,
                 transport: Transport, waypoints: List[Warehouse] = None):
        self.id = id
        self.cargo = cargo
        self.origin = origin
        self.destination = destination
        self.waypoints = waypoints or []
        self.transport = transport
        self.status = "pending"
        self.cost: Optional[float] = None
        self.estimated_time: Optional[timedelta] = None
        self.actual_departure: Optional[datetime] = None
        self.actual_arrival: Optional[datetime] = None

    def calculate_delivery_details(self):
        calculator = CostCalculator()
        distance = self._calculate_total_distance()

        self.cost = calculator.calculate_cost(self.transport, self.cargo, distance)
        self.estimated_time = calculator.calculate_time(distance, self.transport.speed)

    def _calculate_total_distance(self) -> float:
        calculator = DistanceCalculator()
        total_distance = 0.0

        if self.waypoints:
            total_distance += calculator.calculate_distance(self.origin, self.waypoints[0])

            for i in range(len(self.waypoints) - 1):
                total_distance += calculator.calculate_distance(self.waypoints[i], self.waypoints[i + 1])

            total_distance += calculator.calculate_distance(self.waypoints[-1], self.destination)
        else:
            total_distance = calculator.calculate_distance(self.origin, self.destination)

        return total_distance

    def start_delivery(self):
        self.status = "in_transit"
        self.actual_departure = datetime.now()

    def complete_delivery(self):
        self.status = "delivered"
        self.actual_arrival = datetime.now()

    def get_route_info(self) -> str:
        route = [self.origin.name]
        route.extend([wp.name for wp in self.waypoints])
        route.append(self.destination.name)
        return " -> ".join(route)

    def __str__(self):
        return f"Доствка #{self.id}: {self.cargo} по маршруту {self.get_route_info()}"


class TransportationSystem:
    def __init__(self):
        self.warehouses: Dict[int, Warehouse] = {}
        self.transports: Dict[int, Transport] = {}
        self.cargos: Dict[int, Cargo] = {}
        self.deliveries: Dict[int, Delivery] = {}
        self.delivery_counter = 1

    def add_warehouse(self, warehouse: Warehouse):
        self.warehouses[warehouse.id] = warehouse

    def add_transport(self, transport: Transport):
        self.transports[transport.id] = transport

    def add_cargo(self, cargo: Cargo):
        self.cargos[cargo.id] = cargo

    def create_delivery(self, cargo_id: int, origin_id: int, destination_id: int,
                        transport_id: int, waypoint_ids: List[int] = None) -> Delivery:
        cargo = self.cargos.get(cargo_id)
        origin = self.warehouses.get(origin_id)
        destination = self.warehouses.get(destination_id)
        transport = self.transports.get(transport_id)

        if not all([cargo, origin, destination, transport]):
            raise ValueError("Неверные ID объектов")

        if not transport.can_transport(cargo):
            raise ValueError("Выбранный транспорт не может перевозить данный груз")

        waypoints = []
        if waypoint_ids:
            for wp_id in waypoint_ids:
                warehouse = self.warehouses.get(wp_id)
                if warehouse:
                    waypoints.append(warehouse)

        delivery = Delivery(self.delivery_counter, cargo, origin, destination, transport, waypoints)
        delivery.calculate_delivery_details()

        self.deliveries[self.delivery_counter] = delivery
        self.delivery_counter += 1

        return delivery

    def find_suitable_transports(self, cargo: Cargo) -> List[Transport]:
        suitable = []
        for transport in self.transports.values():
            if transport.can_transport(cargo):
                suitable.append(transport)
        return suitable

    def get_delivery_status(self, delivery_id: int) -> Optional[Dict]:
        delivery = self.deliveries.get(delivery_id)
        if not delivery:
            return None

        return {
            'id': delivery.id,
            'status': delivery.status,
            'cost': delivery.cost,
            'estimated_time': delivery.estimated_time,
            'route': delivery.get_route_info(),
            'actual_departure': delivery.actual_departure,
            'actual_arrival': delivery.actual_arrival
        }

    def start_delivery(self, delivery_id: int):
        delivery = self.deliveries.get(delivery_id)
        if delivery:
            delivery.start_delivery()
            return True
        return False

    def complete_delivery(self, delivery_id: int):
        delivery = self.deliveries.get(delivery_id)
        if delivery:
            delivery.complete_delivery()
            return True
        return False

    def get_all_deliveries(self) -> List[Delivery]:
        return list(self.deliveries.values())

    def get_deliveries_by_status(self, status: str) -> List[Delivery]:
        return [d for d in self.deliveries.values() if d.status == status]


def main():
    system = TransportationSystem()

    warehouse1 = Warehouse(1, "Склад Гомель", "Гомель, ул. Ленина 1")
    warehouse2 = Warehouse(2, "Склад Минск", "Минск, Независимости пр. 1")
    warehouse3 = Warehouse(3, "Склад Брест", "Брест, ул. Большая 5")

    system.add_warehouse(warehouse1)
    system.add_warehouse(warehouse2)
    system.add_warehouse(warehouse3)

    truck1 = Truck(1, "Грузовик Volvo", 5000, 30)
    truck2 = Truck(2, "Фургон Mercedes", 2000, 15)
    fridge_truck = RefrigeratorTruck(3, "Рефрижератор Scania", 4000, 25, -20)

    system.add_transport(truck1)
    system.add_transport(truck2)
    system.add_transport(fridge_truck)

    cargo1 = Cargo(1, "Электроника", 500, 5, CargoType.FRAGILE, fragile=True)
    cargo2 = Cargo(2, "Замороженные продукты", 1000, 8, CargoType.REFRIGERATED)
    cargo3 = Cargo(3, "Строительные материалы", 3000, 12, CargoType.GENERAL)

    system.add_cargo(cargo1)
    system.add_cargo(cargo2)
    system.add_cargo(cargo3)

    try:
        print("1.Прямая доставка:")
        delivery1 = system.create_delivery(1, 1, 2, 1)
        print(f"{delivery1}")
        print(f"Стоимость: {delivery1.cost} руб.")
        print(f"Время доставки: {delivery1.estimated_time}")
        print(f"Расстояние: {delivery1._calculate_total_distance():.1f} км")

        print("\n2.Мульти-доставка:")
        delivery2 = system.create_delivery(2, 1, 2, 3, [3])
        print(f"{delivery2}")
        print(f"Стоимость: {delivery2.cost} руб.")
        print(f"Время доставки: {delivery2.estimated_time}")
        print(f"Расстояние: {delivery2._calculate_total_distance():.1f} км")

        print("\n3.Поиск транспорта:")
        suitable = system.find_suitable_transports(cargo2)
        print(f"Подходщий транспорт для '{cargo2.name}':")
        for transport in suitable:
            print(f"     - {transport}")

        print("\n4. Управление доставкой:")
        system.start_delivery(1)
        status = system.get_delivery_status(1)
        print(f"Статус доставки #1: {status['status']}")
        print(f"Маршрут: {status['route']}")

        print("\n5.Все доставки в системе:")
        for delivery in system.get_all_deliveries():
            print(f"#{delivery.id}: {delivery.cargo} -> {delivery.get_route_info()} ({delivery.status})")

    except ValueError as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()