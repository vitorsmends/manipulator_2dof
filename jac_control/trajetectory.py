import pygame
import numpy as np
import math

# --- Configurações da Simulação ---
WIDTH, HEIGHT = 1000, 800
BASE_X, BASE_Y = WIDTH // 2, HEIGHT // 2
L1 = 200  # Comprimento do primeiro elo
L2 = 150  # Comprimento do segundo elo
DT = 0.01        # Passo de tempo da simulação (menor para melhor rastreamento)
KP = 5.0         # Ganho proporcional (pode ser maior com feed-forward)
LAMBDA_SQ = 0.05**2 # Fator de amortecimento DLS

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)

# --- Configurações das Trajetórias ---
TRAJ_CENTER_X = 150.0
TRAJ_CENTER_Y = 100.0
CIRCLE_RADIUS = 100.0
SQUARE_HALF_SIDE = 100.0
TRAJECTORY_PERIOD = 10.0 # Segundos para completar um ciclo

# --- Funções de Cinemática (iguais) ---

def forward_kinematics(q1, q2):
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return (x1, y1), (x2, y2)

def get_jacobian(q1, q2):
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    jacobian = np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [ L1 * c1 + L2 * c12,  L2 * c12]
    ])
    return jacobian

# --- Funções Auxiliares (iguais) ---

def to_pygame_coords(x, y):
    return int(BASE_X + x), int(BASE_Y - y)

def from_pygame_coords(px, py):
    return float(px - BASE_X), float(BASE_Y - py)

# --- Geradores de Trajetória ---

def generate_circular_trajectory(sim_time):
    """
    Gera a posição (x_d) e velocidade (x_dot_d) desejadas para um círculo.
    """
    omega = 2.0 * np.pi / TRAJECTORY_PERIOD
    angle = omega * sim_time
    
    # Posição desejada
    x_d = TRAJ_CENTER_X + CIRCLE_RADIUS * np.cos(angle)
    y_d = TRAJ_CENTER_Y + CIRCLE_RADIUS * np.sin(angle)
    
    # Velocidade desejada (derivada da posição)
    x_dot_d = -CIRCLE_RADIUS * omega * np.sin(angle)
    y_dot_d =  CIRCLE_RADIUS * omega * np.cos(angle)
    
    return np.array([x_d, y_d]), np.array([x_dot_d, y_dot_d])

def generate_square_trajectory(sim_time):
    """
    Gera a posição (x_d) e velocidade (x_dot_d) desejadas para um quadrado.
    """
    phase_time = sim_time % TRAJECTORY_PERIOD
    side_time = TRAJECTORY_PERIOD / 4.0
    speed = (2.0 * SQUARE_HALF_SIDE) / side_time
    
    h = SQUARE_HALF_SIDE # half_side
    c_x, c_y = TRAJ_CENTER_X, TRAJ_CENTER_Y
    
    x_d, y_d = 0, 0
    x_dot_d, y_dot_d = 0, 0

    if 0 <= phase_time < side_time:
        # Lado 1: Mover para a esquerda (do canto superior direito para o esquerdo)
        progress = phase_time / side_time
        x_d = c_x + h - (2 * h * progress)
        y_d = c_y + h
        x_dot_d, y_dot_d = -speed, 0
        
    elif side_time <= phase_time < 2 * side_time:
        # Lado 2: Mover para baixo
        progress = (phase_time - side_time) / side_time
        x_d = c_x - h
        y_d = c_y + h - (2 * h * progress)
        x_dot_d, y_dot_d = 0, -speed

    elif 2 * side_time <= phase_time < 3 * side_time:
        # Lado 3: Mover para a direita
        progress = (phase_time - 2 * side_time) / side_time
        x_d = c_x - h + (2 * h * progress)
        y_d = c_y - h
        x_dot_d, y_dot_d = speed, 0
        
    else: # 3 * side_time <= phase_time < 4 * side_time
        # Lado 4: Mover para cima
        progress = (phase_time - 3 * side_time) / side_time
        x_d = c_x + h
        y_d = c_y - h + (2 * h * progress)
        x_dot_d, y_dot_d = 0, speed

    return np.array([x_d, y_d]), np.array([x_dot_d, y_dot_d])

# --- Função de Desenho Auxiliar ---

def draw_reference_trajectories(screen):
    # Desenha o círculo de referência
    circle_center_screen = to_pygame_coords(TRAJ_CENTER_X, TRAJ_CENTER_Y)
    pygame.draw.circle(screen, GRAY, circle_center_screen, int(CIRCLE_RADIUS), 1)
    
    # Desenha o quadrado de referência
    h = SQUARE_HALF_SIDE
    c_x, c_y = TRAJ_CENTER_X, TRAJ_CENTER_Y
    p1 = to_pygame_coords(c_x + h, c_y + h)
    p2 = to_pygame_coords(c_x - h, c_y + h)
    p3 = to_pygame_coords(c_x - h, c_y - h)
    p4 = to_pygame_coords(c_x + h, c_y - h)
    pygame.draw.line(screen, GRAY, p1, p2, 1)
    pygame.draw.line(screen, GRAY, p2, p3, 1)
    pygame.draw.line(screen, GRAY, p3, p4, 1)
    pygame.draw.line(screen, GRAY, p4, p1, 1)

# --- Função Principal da Simulação ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Controle de Trajetória 2-DOF (DLS + Feed-forward)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # --- Estado do Robô ---
    q_current = np.array([np.pi / 2, np.pi / 4]) 
    
    # --- Controle da Simulação ---
    sim_time = 0.0
    trajectory_mode = 'IDLE' # 'IDLE', 'CIRCLE', 'SQUARE'
    
    # Posições e velocidades desejadas (serão atualizadas pelo gerador)
    target_pos = np.array([0.0, 0.0])
    target_vel = np.array([0.0, 0.0])

    running = True
    while running:
        # --- Processamento de Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    trajectory_mode = 'CIRCLE'
                    sim_time = 0.0 # Reseta o tempo ao trocar de trajetória
                elif event.key == pygame.K_s:
                    trajectory_mode = 'SQUARE'
                    sim_time = 0.0
                elif event.key == pygame.K_i:
                    trajectory_mode = 'IDLE'

        # --- Etapa de Geração de Trajetória ---
        if trajectory_mode == 'CIRCLE':
            sim_time += DT
            target_pos, target_vel = generate_circular_trajectory(sim_time)
        elif trajectory_mode == 'SQUARE':
            sim_time += DT
            target_pos, target_vel = generate_square_trajectory(sim_time)
        else: # 'IDLE'
            # Mantém o alvo na última posição do efetuador
            _, ee_pos = forward_kinematics(q_current[0], q_current[1])
            target_pos = np.array(ee_pos)
            target_vel = np.array([0.0, 0.0])

        # --- Etapa de Controle (Sempre ativa para rastreamento) ---
        
        # 1. Calcular a posição atual (FK)
        j1_pos, ee_pos = forward_kinematics(q_current[0], q_current[1])
        current_pos = np.array(ee_pos)
        
        # 2. Calcular o erro de posição
        error = target_pos - current_pos
        
        # 3. Lei de Controle: Feed-forward (target_vel) + Feedback (Kp * error)
        ee_velocity_ref = target_vel + (KP * error)
        
        # 4. Calcular o Jacobiano
        J = get_jacobian(q_current[0], q_current[1])
        J_T = J.T
        
        # 5. Resolver DLS
        try:
            identity_matrix = np.identity(2)
            inv_term = np.linalg.inv(J @ J_T + LAMBDA_SQ * identity_matrix)
            J_dls = J_T @ inv_term
            
            # 6. Calcular velocidades das juntas
            dq = J_dls @ ee_velocity_ref
            
            # 7. Integrar para atualizar as juntas
            q_current += dq * DT
            
        except np.linalg.LinAlgError:
            print("Erro de Álgebra Linear")

        # --- Etapa de Desenho ---
        screen.fill(BLACK)
        
        # Desenhar instruções
        text_str = f"Modo: {trajectory_mode} (Pressione 'C', 'S', 'I')"
        help_text = font.render(text_str, True, WHITE)
        screen.blit(help_text, (10, 10))
        
        # Desenhar trajetórias de referência
        draw_reference_trajectories(screen)
        
        # Desenhar o alvo (ponto vermelho se movendo)
        target_screen_pos = to_pygame_coords(target_pos[0], target_pos[1])
        pygame.draw.circle(screen, RED, target_screen_pos, 10, 2)
        
        # Desenhar o robô
        base_screen = to_pygame_coords(0, 0)
        j1_screen = to_pygame_coords(j1_pos[0], j1_pos[1])
        ee_screen = to_pygame_coords(ee_pos[0], ee_pos[1])
        
        pygame.draw.line(screen, WHITE, base_screen, j1_screen, 8)
        pygame.draw.line(screen, WHITE, j1_screen, ee_screen, 8)
        pygame.draw.circle(screen, BLUE, base_screen, 12)
        pygame.draw.circle(screen, BLUE, j1_screen, 10)
        pygame.draw.circle(screen, GREEN, ee_screen, 10)
        
        pygame.display.flip()
        clock.tick(1.0 / DT) # Tenta rodar na taxa de simulação

    pygame.quit()

if __name__ == "__main__":
    main()