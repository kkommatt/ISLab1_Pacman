import enum
import sys

import numpy as np
import networkx
import pygame
import random

# Константи
MAZE_CELL_SIZE = 20
MAZE_WIDTH = 30
MAZE_HEIGHT = 30
MAX_START_SCORE = 200
MAX_LEVEL = 5


# Визначаємо кольори у вигляді enum
class Colors(enum.Enum):
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 155)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    PINK = (255, 105, 180)
    CYAN = (0, 255, 255)
    ORANGE = (255, 165, 0)
    WHITE = (255, 255, 255)


# Класи відповідно для Пакмена ти Привидів
class Pacman:
    def __init__(self, position):
        self.position = position

    def move(self, new_position):
        self.position = new_position


class Ghost:
    def __init__(self, position, graph):
        self.position = position
        self.graph = graph

    def move(self, new_position):
        self.position = new_position

    def bfs(self, pacman_position):
        # Спроба знайти найкоротший шлях від поточної позиції привида до позиції Пакмена
        try:
            path = networkx.shortest_path(self.graph, self.position, pacman_position)
        except networkx.NetworkXNoPath:
            path = []
        if path and len(path) > 1:
            return path[1]

        # Якщо шлях не знайдено, то повертаємо сусідню клітинку
        return random.choice(list(self.graph.neighbors(self.position))) if list(
            self.graph.neighbors(self.position)) else self.position


# Генерація лабіринту за допомогою DFS, з урахуванням зростання рівню гри
def dfs(maze_width, maze_height, difficulty_level):
    # Створюємо масив висотаХширина, який представляє лабіринт, 0 - стіна
    maze = np.zeros((maze_height, maze_width), dtype=int)

    # Створюємо прохід для парних індексів - простір розділений стінами, 1 - пустота
    for row in range(1, maze_height - 1, 2):
        for col in range(1, maze_width - 1, 2):
            maze[row, col] = 1

            # Стек для шляху
    path_stack = [(1, 1)]
    # Сет для того, щоб у пройдені клітинки не заходити повторно
    visited_cells = set()

    # Поки клітинки в стеку шляху
    while path_stack:
        # Остання додана в стек клітинка
        current_cell = path_stack[-1]
        # Позначаємо як пройдену
        visited_cells.add(current_cell)

        # Сусідні клітинки, куди ми можемо піти
        available_neighbors_cells = []

        # Ітеруємо по кожному напрямку (2 - 1 (пустота) + 1 (стіна)) вгору, вниз, вліво, вправо
        for direction in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            # Сусідня клітина для поточної з урахуванням напрямку
            neighbor_cell = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            # Перевірка на межі, чи не повторний прохід та чи це пустота
            if (0 <= neighbor_cell[0] < maze.shape[0]
                    and 0 <= neighbor_cell[1] < maze.shape[1]
                    and neighbor_cell not in visited_cells
                    and maze[neighbor_cell] == 1):
                # Якщо всі умови виконані, то додаємо до сусідніх
                available_neighbors_cells.append(neighbor_cell)

        # Якщо є сусідні клітини
        if available_neighbors_cells:
            # Вибираємо одну з
            next_cell = random.choice(available_neighbors_cells)
            # Беремо рандомну сусідню клітину і прибираємо стіну між ними
            maze[(current_cell[0] + next_cell[0]) // 2, (current_cell[1] + next_cell[1]) // 2] = 1
            # Додаємо цю випадково обрану клітину в стек клітин
            path_stack.append(next_cell)
        else:
            # Натомість якщо немає сусідніх клітин, які не на межі, не проходяться повторно, чи є стіною, то беремо з стека
            path_stack.pop()
    # Ускладнюємо лабіринт відповідно до рівню
    if difficulty_level > 1:
        for i in range(maze_width * 5):
            # Випадковим чином у межах обираємо рядок та колонку
            random_row = random.randint(1, maze_height - 2)
            random_col = random.randint(1, maze_width - 2)
            # Якщо ми потрапляємо на стіну, то перетворюємо на пустоту
            maze[random_row, random_col] = 1 if maze[random_row, random_col] == 0 else 1

    # Повертаємо згенерований масив, який задає лабіринт
    return maze


def init_graph(maze_array):
    # Ініціалізація графу
    graph = networkx.Graph()

    # Ітерацію по кожній клітині лабіринту
    for row in range(maze_array.shape[0]):
        for col in range(maze_array.shape[1]):
            # Якщо клітинка пуста, то додаємо в граф, як вузол з координатами
            if maze_array[row, col] == 1:
                graph.add_node((row, col))
                # Перевірка по напрямкам - вгору, вниз, вліво, вправо
                for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    # Переходимо до сусідньої клітини
                    neighbor_row, neighbor_col = row + direction[0], col + direction[1]
                    # Перевіряємо чи сусідня клітинка в межах та чи це теж простір (не стіна)
                    if (0 <= neighbor_row < maze_array.shape[0]
                            and 0 <= neighbor_col < maze_array.shape[1]
                            and maze_array[neighbor_row, neighbor_col] == 1):
                        # Якщо не стіна та лежить в межах, то додаємо як ребро до графу
                        graph.add_edge((row, col), (neighbor_row, neighbor_col))

    # Повертаємо створений граф, який задає лабіринт
    return graph


class GameProcess:
    def __init__(self):
        pygame.font.init()
        self.maze = None
        self.graph = None
        self.pacman = None
        self.ghosts = None
        self.running = None
        self.game_over = None
        self.total_score = None
        self.level_score = None
        self.coins = None
        self.speed = None
        self.counter = None
        self.screen = None
        self.sheight = None
        self.swidth = None
        self.height = None
        self.width = None
        self.current_level = None
        self.start_game()

    def start_game(self):
        self.current_level = 1
        self.width = MAZE_WIDTH
        self.height = MAZE_HEIGHT
        self.swidth = self.width * MAZE_CELL_SIZE
        self.sheight = self.height * MAZE_CELL_SIZE + 20
        self.screen = pygame.display.set_mode((self.swidth, self.sheight))
        pygame.display.set_caption("Pacman")
        self.maze = dfs(self.width, self.height, self.current_level)
        self.graph = init_graph(self.maze)
        self.pacman = Pacman(random.choice(list(zip(*np.where(self.maze == 1)))))

        # Cтворення привидів у випадкових пустотах, крім позиції пакмена, обмеження кількість для певного рівня
        self.ghosts = [Ghost(pos, self.graph) for pos in
                       random.sample([cell for cell in zip(*np.where(self.maze == 1)) if cell != self.pacman.position],
                                     min(3, self.current_level + 1))] if min(3, self.current_level + 1) > 0 else []

        # Гра йде чи ні
        self.running = False

        # Гра завершилась чи ні
        self.game_over = False
        self.total_score = 0
        self.level_score = 0

        # Створює набір монеток випадковим чином
        self.coins = set(random.sample(list(zip(*np.where(self.maze == 1))), len(list(zip(*np.where(self.maze == 1))))))

        # Стартова швидкість привида
        self.speed = 0.25

        # Лічильник пройденого шляху привидом для дотримання визначеної швидкості
        self.counter = 0

    def next_level(self):
        # Відповідним чином зростають параметри для зростання складності гри
        self.current_level += 1
        self.width += 2
        self.height += 2
        self.swidth = self.width * MAZE_CELL_SIZE
        self.sheight = self.height * MAZE_CELL_SIZE + 20
        self.screen = pygame.display.set_mode((self.swidth, self.sheight))
        self.maze = dfs(self.width, self.height, self.current_level)
        self.graph = init_graph(self.maze)
        self.pacman = Pacman(random.choice(list(zip(*np.where(self.maze == 1)))))

        self.ghosts = [Ghost(pos, self.graph) for pos in
                       random.sample([cell for cell in zip(*np.where(self.maze == 1)) if cell != self.pacman.position],
                                     max(3, self.current_level + 1))] if min(3, self.current_level + 1) > 0 else []
        self.coins = set(random.sample(list(zip(*np.where(self.maze == 1))), len(list(zip(*np.where(self.maze == 1))))))
        self.level_score = 0

        self.speed = 0.25 + 0.05 * (self.current_level - 1)

        # Частина коду, що відповідає за відображення для росту рівню складності
        self.screen.fill(Colors.BLACK.value)
        text = pygame.font.Font(None, 30).render("Press space to new level", True, Colors.WHITE.value)
        self.screen.blit(text, (self.swidth // 2 - text.get_width() // 2, 350))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    waiting = False

    # Обробка натискання клавіш для керування Пакменом
    def key_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            keys = pygame.key.get_pressed()
            direction = None

            # Напрямок відповідно до натиснутої клавіші
            if keys[pygame.K_LEFT]:
                direction = (0, -1)
            elif keys[pygame.K_RIGHT]:
                direction = (0, 1)
            elif keys[pygame.K_UP]:
                direction = (-1, 0)
            elif keys[pygame.K_DOWN]:
                direction = (1, 0)
            elif keys[pygame.K_q]:
                pygame.quit()
                sys.exit()

            # Змінюємо координати Пакмена відповідно до натиснутих клавіш і якщо натиснуто
            if direction:
                new_position = (self.pacman.position[0] + direction[0],
                                self.pacman.position[1] + direction[1])
                if (0 <= new_position[0] < self.height
                        and 0 <= new_position[1] < self.width
                        and self.maze[new_position[0], new_position[1]] == 1):
                    self.pacman.move(new_position)

            # Пакмен збирає монетку
            if self.pacman.position in self.coins:
                self.coins.remove(self.pacman.position)
                self.total_score += 10
                self.level_score += 10
                if self.level_score >= MAX_START_SCORE:
                    self.next_level()

    def ghosts_movement(self):
        self.counter += self.speed

        if self.counter >= 1:
            self.counter -= 1

            for ghost in self.ghosts:
                # Знаходимо найближчу допустиму позицію методом BFS у графі
                position_by_bfs = ghost.bfs(self.pacman.position)

                # Якщо валідна до умов знаходження в межах та є простором, а не стіною, то рухаємо привида
                if (0 <= position_by_bfs[0] < self.height
                        and 0 <= position_by_bfs[1] < self.width
                        and self.maze[position_by_bfs[0], position_by_bfs[1]] == 1):
                    ghost.move(position_by_bfs)

    def draw_maze(self):
        self.screen.fill(Colors.BLACK.value)

        # Проходимо по кожній клітині, якщо стіна, то заповнюємо блоком, у якого сторони червоного кольору
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y, x] == 0:
                    # Клітина із стіною - чорний квадрат
                    pygame.draw.rect(self.screen, Colors.BLACK.value,
                                     pygame.Rect(x * MAZE_CELL_SIZE, y * MAZE_CELL_SIZE, MAZE_CELL_SIZE,
                                                 MAZE_CELL_SIZE))
                    # Сторони фарбуються в червоний
                    pygame.draw.rect(self.screen, Colors.RED.value,
                                     pygame.Rect(x * MAZE_CELL_SIZE, y * MAZE_CELL_SIZE, MAZE_CELL_SIZE,
                                                 MAZE_CELL_SIZE), width=1)

                # Якщо простір - фарбуємо в чорний квадрат
                else:
                    pygame.draw.rect(self.screen, Colors.BLACK.value,
                                     pygame.Rect(x * MAZE_CELL_SIZE, y * MAZE_CELL_SIZE, MAZE_CELL_SIZE,
                                                 MAZE_CELL_SIZE))
        # Фарбуємо монетки в білий кольор
        for coin in self.coins:
            pygame.draw.circle(self.screen, Colors.WHITE.value,
                               (coin[1] * MAZE_CELL_SIZE + MAZE_CELL_SIZE // 2,
                                coin[0] * MAZE_CELL_SIZE + MAZE_CELL_SIZE // 2),
                               MAZE_CELL_SIZE // 8)

        # За координатами малюємо пакмена - в жовтий колір
        pygame.draw.circle(self.screen, Colors.YELLOW.value,
                           (self.pacman.position[1] * MAZE_CELL_SIZE + MAZE_CELL_SIZE // 2,
                            self.pacman.position[0] * MAZE_CELL_SIZE + MAZE_CELL_SIZE // 2),
                           MAZE_CELL_SIZE // 2)
        # Відображаємо привидів - блакитний колір
        for ghost in self.ghosts:
            pygame.draw.circle(self.screen, Colors.BLUE.value,
                               (ghost.position[1] * MAZE_CELL_SIZE + MAZE_CELL_SIZE // 2,
                                ghost.position[0] * MAZE_CELL_SIZE + MAZE_CELL_SIZE // 2),
                               MAZE_CELL_SIZE // 2)
        # Інформація про поточний рівень та сумарний рахунок, відображаємо знизу
        score_text = pygame.font.Font(None, 30).render(f"Score: {self.total_score}", True, Colors.WHITE.value)
        level_text = pygame.font.Font(None, 30).render(f"Level: {self.current_level}", True, Colors.WHITE.value)
        self.screen.blit(score_text, (10, self.height * MAZE_CELL_SIZE + 5))
        self.screen.blit(level_text, (self.width * MAZE_CELL_SIZE - 100, self.height * MAZE_CELL_SIZE + 5))
        pygame.display.flip()

    def informative_window(self, value1, value2):
        while True:
            # Відображаємо інформацію щодо подальших можливих дій
            self.screen.fill(Colors.BLACK.value)
            self.screen.blit(pygame.font.Font(None, 30).render(value1, True, Colors.WHITE.value),
                             pygame.font.Font(None, 30).render(value1, True, Colors.WHITE.value).get_rect(
                                 center=(self.swidth // 2, 200)))
            self.screen.blit(pygame.font.Font(None, 30).render(value2, True, Colors.WHITE.value),
                             pygame.font.Font(None, 30).render(value2, True, Colors.WHITE.value).get_rect(
                                 center=(self.swidth // 2, 250)))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.start_game()
                        self.game_processing()
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    # Основний цикл для гри
    def run_game(self):
        while True:
            self.informative_window("Press space to start", "Press Q to quit")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    def game_processing(self):
        self.running = True
        self.game_over = False
        # Якщо рівень не перевищує максимальний, то дозволяється його зростання
        # Цей цикл обробляє рух привидів та стик з пакменом
        while self.running and self.current_level <= MAX_LEVEL:
            self.key_input()

            self.ghosts_movement()
            for ghost in self.ghosts:
                if ghost.position == self.pacman.position:
                    self.game_over = True
                    self.running = False

            self.draw_maze()

            # Параметр ліби - частота оновлення
            pygame.time.Clock().tick(5)

        if self.game_over:
            self.informative_window("Press space to start", "Press Q to quit")

        elif self.current_level > MAX_LEVEL:
            self.informative_window("Press space to start", "Press Q to quit")


game = GameProcess()
game.run_game()
