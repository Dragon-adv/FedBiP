# FedBiP: Implementation Context & Architecture

[cite_start]**Paper Title:** FedBiP: Heterogeneous One-Shot Federated Learning with Personalized Latent Diffusion Models [cite: 5]
[cite_start]**Goal:** Implement a One-Shot Federated Learning framework that uses a pretrained Latent Diffusion Model (LDM) to synthesize training data on the server, personalized to client-specific distributions (feature/label heterogeneity) without accessing private data directly. [cite: 52, 53]

---

## 1. System Overview & Data Flow

### 1.1 Components
* **Clients ($K$):** Possess private data $D^k = \{x_i^k, y_i^k\}$. Perform "Bi-Level Personalization" locally.
* **Server:** Receives personalized vectors from clients (One-Shot). Generates synthetic dataset $D_{syn}$. [cite_start]Trains global classification model $\phi$. [cite: 105]
* **Base Models:**
    * **LDM:** Pretrained Stable Diffusion v1-4 (`CompVis/stable-diffusion-v1-4`). [cite_start]Encoder $\mathcal{E}$, Decoder $\mathcal{D}$, UNet $\epsilon_\theta$, Text Encoder $\tau_\theta$. [cite: 207]
    * [cite_start]**Classifier:** ResNet-18 (pretrained on ImageNet). [cite: 207]

### 1.2 The "Bi-Level" Personalization Strategy
FedBiP modifies the generation process in two ways:
1.  [cite_start]**Instance-Level:** Modifying the *initial noise latent* $z(T)$ based on real data to preserve structure. [cite: 54]
2.  [cite_start]**Concept-Level:** Optimizing *textual embedding vectors* (Domain concept $S$ and Category concept $V$) to capture style/domain shifts. [cite: 55]

---

## 2. Methodology & Mathematical Formulation

### 2.1 Client-Side: Instance-Level Personalization
Instead of sampling $z(T) \sim \mathcal{N}(0, I)$, clients compute it from local data to preserve privacy and structure.

**Steps:**
1.  [cite_start]**Encode:** Get latent $z_i(0) = \mathcal{E}(x_i)$. [cite: 156]
2.  **Interpolate (Privacy Preserving):** Mix with another sample $x_{i'}$ from the same class:
    $$\bar{z}_i(0) = \gamma z_i(0) + (1-\gamma) z_{i'}(0)$$
    * Constraint: $i \neq i'$, $y_i = y_{i'}$.
    * [cite_start]$\gamma \sim \mathcal{N}(0.5, 0.1^2)$, clipped to $[0, 1]$. [cite: 160, 162]
3.  **Forward Diffusion (Noise Addition):** Add noise up to timestep $T$:
    $$z_i(T) = \delta(T, \bar{z}_i(0)) = \sqrt{\alpha_T}\bar{z}_i(0) + \sqrt{1-\alpha_T}\epsilon$$
    * [cite_start]This $z_i(T)$ is uploaded to the server. [cite: 90, 158]

### 2.2 Client-Side: Concept-Level Personalization
Clients fine-tune textual embeddings to align LDM generation with their local domain (e.g., "sketch" or "medical").

**Prompt Template:**
[cite_start]"A $[S]$ style of a $[V_y]$" (e.g., $[S]$ = domain concept, $[V_y]$ = class concept). [cite: 171]

**Learnable Parameters:**
* [cite_start]$S \in \mathbb{R}^{n_s \times d_w}$: Domain concept vector (shared across all classes in client). [cite: 170]
* [cite_start]$V \in \mathbb{R}^{C \times n_v \times d_w}$: Category concept vectors (one per class). [cite: 170]
* [cite_start]**Frozen:** LDM weights $\epsilon_\theta, \mathcal{E}, \mathcal{D}, \tau_\theta$ are frozen; only $S$ and $V$ are optimized. [cite: 169]

**Optimization Objective (Equation 4):**
$$L_g = \mathbb{E}_{z(t), t, \epsilon} [||\epsilon - \epsilon_\theta(z(t), t, \tau_\theta(S, V_y))||_2^2]$$
* [cite_start]$t \sim \text{Uniform}(\{1, ..., T\})$. [cite: 173]

### 2.3 Server-Side: Synthesis & Training
**Input:** $\{z_i^k(T), y_i^k, S^k, V^k\}$ from all clients.

**Generation Process (Equation 5):**
1.  **Perturb Domain Concept:** $\hat{S}^k = S^k + \eta$, where $\eta \sim \mathcal{N}(0, \sigma_\eta)$. [cite_start]Increases diversity. [cite: 177]
2.  **Denoise:** Start from uploaded noisy latent $z_i^k(T)$ (NOT random noise).
3.  **Decode:**
    $$\tilde{x}_i = \mathcal{D}(\text{Denoise}(z_i^k(T), \tau_\theta(\hat{S}^k, V_{y_i}^k)))$$
    [cite_start][cite: 179]

**Classification Training (Equation 6):**
Train model $\phi$ on synthetic dataset $D_{syn} = \{(\tilde{x}_i, y_i)\}$.
$$L_{cls} = L_{CE}(\phi(\tilde{x}), y)$$
[cite_start][cite: 188]

---

## 3. Algorithm Workflow (Pseudo-code Logic)

### ClientUpdate(k):
1.  Initialize $S^k, V^k$ randomly.
2.  **Instance Loop:** For each image $x_i$ in $D^k$:
    * Select pair $x_{i'}$ (same class).
    * Compute interpolated $\bar{z}(0)$.
    * Compute noisy $z_i(T)$ (Max timestep $T$).
3.  **Concept Loop:** For `local_steps`:
    * Sample batch $(x_b, y_b)$.
    * Compute LDM Loss ($L_g$) using prompt "A $[S^k]$ style of a $[V_{y_b}^k]$".
    * Update $S^k, V^k$.
4.  **Upload:** $\{z_i(T), y_i\}_{i=1}^{N_k}$, $S^k$, $\{V_j^k\}_{j=1}^C$.
[cite_start][cite: 105]

### ServerUpdate:
1.  Initialize classifier $\phi$.
2.  Receive payloads from all clients.
3.  **Generation Loop:** For each uploaded latent $z_i(T)$:
    * Perturb concept $\hat{S}$.
    * Generate image $\tilde{x} = \text{LDM\_Inference}(z_i(T), \text{prompt}(\hat{S}, V_{y_i}))$.
    * Add to $D_{syn}$.
4.  **Training:** Train $\phi$ on $D_{syn}$ to convergence.
[cite_start][cite: 105]

---

## 4. Implementation Details & Hyperparameters

* **Prompt Templates:**
    * [cite_start]DomainNet/PACS: "A $[S]$ style of a $[V_{CLS}]$". [cite: 238]
    * [cite_start]DermaMNIST: "A dermatoscopic image of a $[CLS]$, a type of pigmented skin lesions." [cite: 230]
    * [cite_start]Satellite (UCM): "A centered satellite photo of $[CLS]$." [cite: 231]
* **Optimization:**
    * [cite_start]Optimize concept vectors for **50 epochs** at each client. [cite: 210]
* **Generative Scale:**
    * [cite_start]Can generate 1x, 5x, or 10x the original dataset size (FedBiP-L recommends 10x). [cite: 208, 275]
* **Privacy Parameters:**
    * [cite_start]Latent interpolation $\gamma$: Mean 0.5, Std 0.1. [cite: 162]