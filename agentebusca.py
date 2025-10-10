import sys
import os
from collections import deque
import cv2
import numpy as np

# Environment
class Environment:
    """
    Lê e mantém o mapa (grid).
    Fornece getSensor(pos, dir) que retorna 3x3 com mat[2][2] = direção.
    """
    DIR_DELTAS = {'N': (-1, 0), 'S': (1, 0), 'L': (0, 1), 'O': (0, -1)}

    def __init__(self, file_name='maze.txt', grid_lines=None):
        if file_name:
            self.grid = [list(line.rstrip("\n")) for line in open('maze.txt')]
        elif grid_lines:
            self.grid = [list(line) for line in grid_lines]
        else:
            raise ValueError("Passe um nome de arquivo ou grid lines")

        # normalizar largura das linhas (preencher com 'X' se necessário)
        maxc = max(len(row) for row in self.grid)
        for r in self.grid:
            if len(r) < maxc:
                r += ['X'] * (maxc - len(r))

        self.rows = len(self.grid)
        self.cols = maxc

        # localizar entrada, saída, contar comidas
        self.entrance = None
        self.exit = None
        self.food_count = 0

        for i in range(self.rows):
            for j in range(self.cols):
                c = self.grid[i][j]
                if c == 'E':
                    self.entrance = (i, j)
                elif c == 'S':
                    self.exit = (i, j)
                elif c == 'o':
                    self.food_count += 1

        if self.entrance is None or self.exit is None:
            raise ValueError("Mapa precisa ter 'E' (entrada) e 'S' (saída)")

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_cell(self, pos):
        r, c = pos
        if not self.in_bounds(pos):
            return 'X'
        return self.grid[r][c]

    def set_cell(self, pos, val):
        r, c = pos
        if self.in_bounds(pos):
            self.grid[r][c] = val

    def remove_food(self, pos):
        """Remove comida em pos (se houver). Retorna True se removeu."""
        if self.get_cell(pos) == 'o':
            self.set_cell(pos, '_')
            self.food_count -= 1
            return True
        return False

    def getSensor(self, agent_pos, agent_dir):
        """
        Retorna uma matriz 3x3 de caracteres com a vizinhança absoluta do agente.
        A célula [2][2] (canto inferior direito) recebe o agente_dir.
        """
        sr = [['X'] * 3 for _ in range(3)]
        ar, ac = agent_pos
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr = ar + dr
                cc = ac + dc
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    sr[dr+1][dc+1] = self.grid[rr][cc]
                else:
                    sr[dr+1][dc+1] = 'X'
        sr[2][2] = agent_dir
        return sr

    def try_move(self, agent, new_dir=None):
        """
        Move o agente se a célula à frente não for parede.
        Recebe um objeto agent com atributos pos (r,c) e direction.
        Retorna True se moveu.
        """
        if new_dir is not None:
            agent.direction = new_dir
        dr, dc = Environment.DIR_DELTAS[agent.direction]
        nr, nc = agent.pos[0] + dr, agent.pos[1] + dc
        if not self.in_bounds((nr, nc)):
            return False
        if self.grid[nr][nc] == 'X':
            return False
        agent.pos = (nr, nc)
        return True

# Agent
class Agent:
    ARROW = {'N': '^', 'S': 'v', 'L': '>', 'O': '<'}

    def __init__(self, env: Environment):
        self.env = env
        self.pos = env.entrance
        self.direction = 'N'
        self.memory = {}  # (r,c) -> char visto
        self.steps = 0
        self.collected = 0
        self.frames = []  # lista de frames

        # objetivo: número de comidas a coletar (padrão = número real no env)
        self.target_food = env.food_count

    def sense_and_update(self):
        mat = self.env.getSensor(self.pos, self.direction)
        ar, ac = self.pos
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 1 and dc == 1:
                    continue  # cantinho com a direção
                rr, cc = ar + dr, ac + dc
                if 0 <= rr < self.env.rows and 0 <= cc < self.env.cols:
                    self.memory[(rr, cc)] = mat[dr+1][dc+1]
        return mat

    def set_direction(self, d):
        if d not in ['N', 'S', 'L', 'O']:
            raise ValueError("Direção inválida")
        self.direction = d

    def move_forward(self):
        moved = self.env.try_move(self)
        if moved:
            self.steps += 1
            # se houver comida na nova célula, coleta
            if self.env.remove_food(self.pos):
                self.collected += 1
        return moved

    def neighbors(self, pos):
        for d, (dr, dc) in Environment.DIR_DELTAS.items():
            yield (pos[0]+dr, pos[1]+dc), d

    def bfs(self, start, goal_test, allow_unknown=False):
        """
        BFS simples sobre posições usando self.memory como conhecimento.
        Se allow_unknown=True, trata desconhecido como possível caminho.
        Retorna lista de posições (path) ou None.
        """
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            if goal_test(cur):
                # reconstrói caminho
                path = []
                p = cur
                while p is not None:
                    path.append(p)
                    p = prev[p]
                path.reverse()
                return path
            for (npos, d) in self.neighbors(cur):
                if npos in prev: 
                    continue
                if not self.env.in_bounds(npos):
                    continue
                cell = self.memory.get(npos, None)
                if cell is None and not allow_unknown:
                    continue
                if cell == 'X':
                    continue
                prev[npos] = cur
                q.append(npos)
        return None

    def find_nearest_known_food(self):
        foods = [p for p, ch in self.memory.items() if ch == 'o']
        if not foods:
            return None
        return self.bfs(self.pos, lambda p: p in foods, allow_unknown=False)

    def find_nearest_unknown(self):
        targets = set()
        # qualquer célula desconhecida adjacente a um conhecido livre é alvo de exploração
        for (p, ch) in list(self.memory.items()) + [(self.pos, None)]:
            if ch == 'X':
                continue
            for (npos, d) in self.neighbors(p):
                if not self.env.in_bounds(npos):
                    continue
                if npos not in self.memory:
                    targets.add(npos)
        if not targets:
            return None
        return self.bfs(self.pos, lambda p: p in targets, allow_unknown=False)

    def follow_path(self, path, render_callback=None):
        """
        Segue o caminho (lista de posições); path[0] deve ser a posição atual.
        Após cada passo, atualiza sensor e, se 'render_callback' fornecido, chama callback(frame).
        """
        for i in range(1, len(path)):
            cur = path[i-1]
            nxt = path[i]
            delta = (nxt[0] - cur[0], nxt[1] - cur[1])
            # define direção necessária
            for d, (dr, dc) in Environment.DIR_DELTAS.items():
                if (dr, dc) == delta:
                    self.set_direction(d)
                    break
            moved = self.move_forward()
            self.sense_and_update()
            if render_callback:
                frame = render_callback(self.pos, self.direction)
                self.frames.append(frame)
        return

    def run(self, render_callback=None, max_steps=10000):
        """
        Loop principal do agente:
         - enquanto não coletou target_food, pega comidas conhecidas ou explora
         - ao completar, vai para saída
        render_callback(pos, dir) -> retorna frame numpy para armazenar/exibir.
        """
        # inicial sensing e frame
        self.sense_and_update()
        if render_callback:
            self.frames.append(render_callback(self.pos, self.direction))

        iterations = 0
        while iterations < max_steps:
            iterations += 1

            # se já coletou todas as comidas, vai para saída
            if self.collected >= self.target_food:
                path = self.bfs(self.pos, lambda p: p == self.env.exit, allow_unknown=False)
                if path is None:
                    path = self.bfs(self.pos, lambda p: p == self.env.exit, allow_unknown=True)
                if path:
                    self.follow_path(path, render_callback=render_callback)
                break

            # procurar comida conhecida
            plan = self.find_nearest_known_food()
            if plan:
                self.follow_path(plan, render_callback=render_callback)
                continue

            # explorar célula desconhecida
            plan = self.find_nearest_unknown()
            if plan:
                self.follow_path(plan, render_callback=render_callback)
                continue

            # fallback: se existem comidas no env que ainda não observamos, tentar alcançá-las permitindo unknown
            unseen_foods = []
            for i in range(self.env.rows):
                for j in range(self.env.cols):
                    if self.env.get_cell((i, j)) == 'o' and (i, j) not in self.memory:
                        unseen_foods.append((i, j))
            if unseen_foods:
                path = self.bfs(self.pos, lambda p: p in unseen_foods, allow_unknown=True)
                if path:
                    self.follow_path(path, render_callback=render_callback)
                    continue

            # sem plano: termina (ou poderia ficar fazendo movimentos randômicos)
            break

        # última atualização / frame
        self.sense_and_update()
        if render_callback:
            self.frames.append(render_callback(self.pos, self.direction))

    @property
    def score(self):
        return self.collected * 10 - self.steps


# ---------------------------
# Função de desenho (mapa + agente)
# ---------------------------
def draw_map_frame(env: Environment, agent_pos, agent_dir, cell_size=40):
    """
    Desenha o mapa em uma imagem NumPy (BGR).
    Retorna frame (np.uint8).
    """

    rows, cols = env.rows, env.cols
    height = rows * cell_size
    width = cols * cell_size

    # Frame branco
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Cores BGR
    COLOR_WALL = (0, 0, 0)
    COLOR_PATH = (255, 255, 255)
    COLOR_FOOD = (0, 140, 255)  # laranja
    COLOR_ENTRANCE = (0, 255, 0)
    COLOR_EXIT = (0, 0, 255)
    COLOR_GRID = (200, 200, 200)
    COLOR_AGENT = (255, 0, 255)  # rosa
    COLOR_ARROW = (0, 0, 0)

    # Desenha células
    for i in range(rows):
        for j in range(cols):
            ch = env.grid[i][j]
            x1, y1 = j * cell_size, i * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size

            if ch == 'X':
                color = COLOR_WALL
            elif ch == '_':
                color = COLOR_PATH
            elif ch == 'o':
                color = COLOR_FOOD
            elif ch == 'E':
                color = COLOR_ENTRANCE
            elif ch == 'S':
                color = COLOR_EXIT
            else:
                color = COLOR_PATH

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
            # borda fina
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GRID, thickness=1)

    # Desenha agente no centro da célula
    ax = agent_pos[1] * cell_size + cell_size // 2
    ay = agent_pos[0] * cell_size + cell_size // 2
    radius = int(cell_size * 0.35)
    cv2.circle(frame, (ax, ay), radius, COLOR_AGENT, thickness=-1)

    # seta indicando direção
    # calcular ponta da seta
    arrow_len = int(cell_size * 0.45)
    dx = 0; dy = 0
    if agent_dir == 'N':
        dx, dy = 0, -arrow_len
    elif agent_dir == 'S':
        dx, dy = 0, arrow_len
    elif agent_dir == 'L':
        dx, dy = arrow_len, 0
    elif agent_dir == 'O':
        dx, dy = -arrow_len, 0
    tip = (ax + dx, ay + dy)
    cv2.arrowedLine(frame, (ax, ay), tip, COLOR_ARROW, 2, tipLength=0.3)

    return frame


# ---------------------------
# Main: rodar simulação e salvar vídeo
# ---------------------------
def main(mapfile=None, output_video="simulation.mp4", show_window=False):
    # carregar mapa
    if mapfile and os.path.exists(mapfile):
        env = Environment(filename=mapfile)
    else:
        env = Environment(grid_lines=SAMPLE_MAP)

    agent = Agent(env)

    # função que desenha e retorna frame
    def render_cb(pos, direction):
        return draw_map_frame(env, pos, direction, cell_size=48)

    # roda simulação (o agente chamará render_callback para cada movimento)
    agent.run(render_callback=render_cb)

    # Se não gerou frames (caso estranho), desenhar pelo menos o inicial
    if not agent.frames:
        agent.frames.append(draw_map_frame(env, agent.pos, agent.direction, cell_size=48))

    # criar VideoWriter
    h, w, _ = agent.frames[0].shape
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for frm in agent.frames:
        writer.write(frm)
        if show_window:
            cv2.imshow("Simulação", frm)
            # pausinha para visualização interativa
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    writer.release()
    if show_window:
        cv2.destroyAllWindows()

    # resumo final
    print("Simulação finalizada.")
    print(f"Steps: {agent.steps}")
    print(f"Comidas coletadas: {agent.collected} / {agent.target_food}")
    print(f"Score: {agent.score}")
    print(f"Vídeo salvo em: {output_video}")


# ---------------------------
# Executa quando chamado diretamente
# ---------------------------
if __name__ == "__main__":
    mapfile = sys.argv[1] if len(sys.argv) > 1 else 'maze.txt'
    main(mapfile=mapfile, output_video="simulation.mp4", show_window=False)
