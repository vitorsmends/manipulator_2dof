import pygame
import numpy as np
import math
import matplotlib.pyplot as plt

# --- Configurações da Simulação ---
WIDTH, HEIGHT = 1000, 800
BASE_X, BASE_Y = WIDTH // 2, HEIGHT // 2
L1 = 200  # Comprimento do primeiro elo
L2 = 150  # Comprimento do segundo elo
DT = 0.01        # Passo de tempo da simulação
KP = 5.0         # Ganho proporcional
LAMBDA_SQ = 0.05**2 # Fator de amortecimento DLS
SIMULATION_DURATION = 20.0 # Duração total da simulação em segundos

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)

# --- Configurações da Trajetória Circular ---
TRAJ_CENTER_X = 150.0
TRAJ_CENTER_Y = 100.0
CIRCLE_RADIUS = 100.0
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

# --- Gerador de Trajetória Circular ---

def generate_circular_trajectory(sim_time):
    omega = 2.0 * np.pi / TRAJECTORY_PERIOD
    angle = omega * sim_time
    
    x_d = TRAJ_CENTER_X + CIRCLE_RADIUS * np.cos(angle)
    y_d = TRAJ_CENTER_Y + CIRCLE_RADIUS * np.sin(angle)
    x_dot_d = -CIRCLE_RADIUS * omega * np.sin(angle)
    y_dot_d =  CIRCLE_RADIUS * omega * np.cos(angle)
    
    return np.array([x_d, y_d]), np.array([x_dot_d, y_dot_d])

# --- Função de Plotagem ---

def plot_results(logs):
    """
    Gera os gráficos de controle usando matplotlib.
    """
    # Converter listas para arrays numpy para facilitar a plotagem
    log_time = np.array(logs['time'])
    log_xd = np.array(logs['xd'])
    log_yd = np.array(logs['yd'])
    log_x = np.array(logs['x'])
    log_y = np.array(logs['y'])
    log_error_mag = np.array(logs['error_mag'])

    plt.figure(figsize=(18, 6))

    # Gráfico 1: Rastreamento de Posição (X vs Y)
    plt.subplot(1, 3, 1)
    plt.plot(log_xd, log_yd, 'r--', label='Trajetória Desejada')
    plt.plot(log_x, log_y, 'b-', label='Trajetória Atual')
    plt.title('Rastreamento de Posição (X vs Y)')
    plt.xlabel('Posição X (pixels)')
    plt.ylabel('Posição Y (pixels)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Gráfico 2: Componentes da Posição vs. Tempo
    plt.subplot(1, 3, 2)
    plt.plot(log_time, log_xd, 'r--', label='X Desejado')
    plt.plot(log_time, log_x, 'r', label='X Atual')
    plt.plot(log_time, log_yd, 'b--', label='Y Desejado')
    plt.plot(log_time, log_y, 'b', label='Y Atual')
    plt.title('Componentes da Posição vs. Tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição (pixels)')
    plt.legend()
    plt.grid(True)

    # Gráfico 3: Magnitude do Erro vs. Tempo
    plt.subplot(1, 3, 3)
    plt.plot(log_time, log_error_mag, 'k-')
    plt.title('Magnitude do Erro vs. Tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro ||e|| (pixels)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- Função Principal da Simulação ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Controle de Trajetória - CÍRCULO")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # --- Estado do Robô ---
    q_current = np.array([np.pi / 2, np.pi / 4]) 
    
    # --- Logging ---
    sim_time = 0.0
    logs = {
        'time': [], 'xd': [], 'yd': [], 'x': [], 'y': [], 'error_mag': []
    }

    running = True
    while running and sim_time <= SIMULATION_DURATION:
        
        # --- Processamento de Eventos (para poder fechar a janela) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        sim_time += DT

        # --- Geração de Trajetória ---
        target_pos, target_vel = generate_circular_trajectory(sim_time)
        
        # --- Etapa de Controle ---
        j1_pos, ee_pos = forward_kinematics(q_current[0], q_current[1])
        current_pos = np.array(ee_pos)
        
        error = target_pos - current_pos
        ee_velocity_ref = target_vel + (KP * error)
        
        J = get_jacobian(q_current[0], q_current[1])
        J_T = J.T
        
        try:
            identity_matrix = np.identity(2)
            inv_term = np.linalg.inv(J @ J_T + LAMBDA_SQ * identity_matrix)
            J_dls = J_T @ inv_term
            dq = J_dls @ ee_velocity_ref
            q_current += dq * DT
        except np.linalg.LinAlgError:
            print("Erro de Álgebra Linear")
            running = False
            
        # --- Logging dos dados ---
        logs['time'].append(sim_time)
        logs['xd'].append(target_pos[0])
        logs['yd'].append(target_pos[1])
        logs['x'].append(current_pos[0])
        logs['y'].append(current_pos[1])
        logs['error_mag'].append(np.linalg.norm(error))

        # --- Etapa de Desenho ---
        screen.fill(BLACK)
        
        text_str = f"Tempo: {sim_time:.2f}s / {SIMULATION_DURATION:.0f}s"
        help_text = font.render(text_str, True, WHITE)
        screen.blit(help_text, (10, 10))

        # Desenhar trajetória de referência
        circle_center_screen = to_pygame_coords(TRAJ_CENTER_X, TRAJ_CENTER_Y)
        pygame.draw.circle(screen, GRAY, circle_center_screen, int(CIRCLE_RADIUS), 1)
        
        # Desenhar o alvo
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
        pygame.draw.circle(screen, GREEN, ee_screen, 1)