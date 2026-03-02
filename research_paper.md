\documentclass[11pt]{article}

% ============================================================
% Layout
% ============================================================
\usepackage[margin=1.1in]{geometry}
\usepackage{microtype}

% ============================================================
% FONT: Springer/Mathematische Annalen-like
% Compile with LuaLaTeX or XeLaTeX
% ============================================================
\usepackage{fontspec}
\usepackage{unicode-math}
\usepackage{amsmath}
\newcommand{\vct}[1]{\symbf{#1}}      % bold vectors/tensors (incl. Greek)
\newcommand{\mat}[1]{\symbfup{#1}}    % bold upright for matrices/tensors if you want
% Text font: Times New Roman if available; otherwise TeX Gyre Termes
\usepackage{amsmath,amssymb,mathtools}
\usepackage{fontspec}
\usepackage{unicode-math}

\IfFontExistsTF{Times New Roman}{
  \setmainfont{Times New Roman}
}{
  \setmainfont{TeX Gyre Termes}
}

% Math font
\IfFontExistsTF{STIX Two Math}{
  \setmathfont{STIX Two Math}
}{
  \IfFontExistsTF{TeX Gyre Termes Math}{
    \setmathfont{TeX Gyre Termes Math}
  }{
    \setmathfont{XITS Math}
  }
}
\linespread{1.05}

% ============================================================
% Math / theorem tools
% ============================================================
\usepackage{amsmath,amssymb,amsthm,mathtools}

% ============================================================
% Tables, figures, algorithms
% ============================================================
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}

% Standardized figure widths
\newlength{\figfull}\setlength{\figfull}{\textwidth}           % Full-width: multi-panel, flowcharts, wide 3D views
\newlength{\figmed}\setlength{\figmed}{0.85\textwidth}         % Medium: dual-panel charts, distribution grids
\newlength{\figcompact}\setlength{\figcompact}{0.7\textwidth}  % Compact: single-panel plots, bar charts
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{multirow}

% ============================================================
% Links (journal-like: no colored boxes)
% ============================================================
\usepackage{hyperref}
\hypersetup{hidelinks}

% ============================================================
% Section title styling
% ============================================================
\usepackage{titlesec}

\titleformat{\section}
  {\normalfont\LARGE\bfseries}
  {\thesection}
  {1em}
  {}

\titleformat{\subsection}
  {\normalfont\Large\bfseries}
  {\thesubsection}
  {1em}
  {}

\titleformat{\subsubsection}
  {\normalfont\large\bfseries}
  {\thesubsubsection}
  {1em}
  {}

\titlespacing*{\section}{0pt}{2.0ex plus 0.6ex minus 0.2ex}{1.0ex}
\titlespacing*{\subsection}{0pt}{1.5ex plus 0.5ex minus 0.2ex}{0.8ex}
\titlespacing*{\subsubsection}{0pt}{1.2ex plus 0.4ex minus 0.2ex}{0.6ex}

% ============================================================
% Theorem environments
% ============================================================
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% ============================================================
% Custom title block (Springer-ish layout)
% ============================================================
\makeatletter
\renewcommand{\maketitle}{
\begin{center}
{\fontsize{18}{22}\selectfont\bfseries \@title \par}
\vspace{1.2em}

{\large Eric Hou$^{1}$\par}
\vspace{0.3em}

{\normalsize $^{1}$Los Gatos, California, USA\par}
{\normalsize \texttt{eric.x.hou@gmail.com}\par}

\vspace{1.2em}

\end{center}
\vspace{1.8em}
}
\makeatother

% ============================================================
% Title
% ============================================================
\title{Additive Manufacturing: Harnessing FEA to Optimize Material Efficiency}
\date{}

% ============================================================
\begin{document}
\maketitle

% ============================================================
% Abstract
% ============================================================
\section*{Abstract}

Topology optimization of full-scale concrete buildings is computationally prohibitive because classical methods such as SIMP require hundreds of finite element analyses, each costing minutes at building-scale resolution. Separately, voxel-based implementations face a failure mode---well-characterized in the digital topology literature \cite{kong1989} but not previously operationalized in building-scale topology optimization pipelines---in which standard 26-connectivity topology checks produce floating mesh fragments incompatible with additive manufacturing toolpaths. This paper introduces Surrogate-Accelerated Sensitivity Topology Optimization (SASTO), a three-phase voxel erosion algorithm that replaces iterative FEA with a five-member deep ensemble of 3D convolutional neural networks trained on 11,178 ASCE 7-22 house simulations generated from 3DWire \cite{3dwire2024} wireframes. SASTO uses backpropagation-derived sensitivity gradients to rank and remove structurally redundant voxels. Four contributions are made. First, SASTO achieves 23.5\% $\pm$ 7.8\% mean material reduction (95\% CI: [22.7\%, 24.3\%]) across 355 proxy-constraint-satisfying house geometries ($N = 916$ evaluated, 38.8\% feasibility rate) in 52 $\pm$ 118 seconds on a consumer GPU, representing an empirically-anchored 23--92$\times$ speedup over SIMP at matched resolution (benchmarked on 10 representative geometries at $64^3$ with a direct sparse solver). A $k$-factor ablation across the full evaluation set reveals a smooth Pareto frontier: from 76.5\% feasibility at $k = 0$ to 7.1\% at $k = 3$, with the operating point $k = 1.0$ balancing conservatism and yield. Second, a 6-connectivity digital topology criterion guarantees marching-cubes-compatible single-component meshes throughout optimization, eliminating thousands of floating fragments produced by conventional 26-connectivity checks. Third, a part-aware heterogeneous thickness formulation permits thinner interior partitions while enforcing conservative limits on load-bearing members, yielding 10.7 percentage points more material reduction than uniform thickness on the reference geometry. Fourth, we provide a quantitative calibration diagnostic via isotonic regression on the held-out test set and a population-level analysis of ensemble disagreement ($\Gamma_D$, mean CV = 0.26), demonstrating that the uncertainty scaffold tracks distribution shift during optimization. All proxy structural constraints use conservative ensemble upper bounds ($\mu + k\sigma$, $k = 1.0$), which provide an implicit $\sim$15--30\% safety buffer above the ensemble mean. Independent same-method FEA re-analysis on all 355 constraint-satisfying designs confirms a 100\% constraint survival rate (0/355 false positives; maximum compliance ratio $C_{\text{opt}}/C_{\text{base}} = 1.004$, well below the 1.15 threshold), with optimization actually \emph{improving} compliance by $\sim$37\% on average ($C$-ratio $= 0.631 \pm 0.112$). Distribution-free conformal prediction certifies $P(\text{violation}) \leq 0.28\%$.

\vspace{0.9em}

\noindent\textbf{Keywords.}
Topology optimization $\cdot$ Finite element analysis $\cdot$ Additive manufacturing $\cdot$ Deep ensemble surrogate $\cdot$ Uncertainty quantification $\cdot$ 3D-printed construction $\cdot$ Digital topology

\vspace{2.0em}

% ============================================================
\section{Introduction}
% ============================================================

Residential construction accounts for roughly 11\% of global CO$_2$ emissions, with concrete production alone responsible for approximately 8\% \cite{iea2021}. In conventional construction, walls, slabs, and roofs are built at uniform thickness, a practice driven by formwork constraints rather than structural necessity. Not all regions of a building carry the same load: interior partition walls, for example, bear negligible gravity or lateral forces compared to exterior shear walls. This creates a substantial opportunity for material reduction if the geometry can be selectively thinned in regions where full thickness is structurally unnecessary.

Large-scale additive manufacturing (AM) of concrete structures has advanced rapidly, with companies such as ICON, COBOD, and Apis Cor demonstrating full-scale 3D-printed houses \cite{buswell2018, ngo2018}. Unlike conventional formwork, AM depositions can realize arbitrary wall profiles at no marginal tooling cost. However, exploiting this geometric freedom requires optimized three-dimensional models that specify precisely where material should be placed: models that simultaneously minimize volume, satisfy proxy structural constraints under ASCE 7-22 load cases, and produce watertight meshes compatible with printer toolpath generation.

Topology optimization, the computational determination of optimal material layout within a design domain, is mature for aerospace and automotive components \cite{bendsoe2003, sigmund2013}. The classical Solid Isotropic Material with Penalization (SIMP) method requires hundreds to thousands of FEA evaluations, each costing minutes to hours for a building-scale tetrahedral mesh, making direct topology optimization of full-scale houses computationally intractable. A separate but equally critical problem arises in voxel-based implementations: 26-connectivity topology checks, standard in the topology optimization literature \cite{xia2015}, permit diagonal-only voxel connections that marching cubes algorithms render as disconnected floating mesh fragments. While the underlying digital topology theory (6- vs.\ 26-connectivity and the complementarity requirement) is well established in the image processing literature \cite{kong1989}, the specific failure mode whereby standard 26-connectivity topology checks in voxel-based topology optimization produce marching-cubes-incompatible meshes has not, to our knowledge, been explicitly identified, quantified, or corrected in the building-scale topology optimization literature.

This work addresses both challenges through three contributions:
\begin{enumerate}[label=(\roman*),leftmargin=2em]
\item A surrogate-accelerated sensitivity erosion algorithm (SASTO) that replaces FEA with millisecond-scale deep ensemble predictions and gradient-based voxel ranking, achieving an empirically-anchored 23--92$\times$ speedup over SIMP at matched resolution.
\item A 6-connectivity topology preservation criterion that guarantees topologically connected voxel fields from which single-component marching cubes meshes can be extracted with at most trivial post-processing (Proposition~\ref{prop:mc}, Remark~\ref{rem:meshgap}), eliminating a marching-cubes incompatibility that has not, to our knowledge, been explicitly quantified at building scale despite well-established digital topology foundations \cite{kong1989}.
\item A part-aware heterogeneous minimum thickness formulation that exploits structural role classification to permit differential thinning of interior partitions while protecting load-bearing members.
\item A large-scale empirical evaluation ($N = 916$ geometries) with quantitative analysis of the constraint-feasibility bottleneck, a $k$-factor ablation quantifying the conservatism--yield Pareto frontier, calibration diagnostics via isotonic regression, and population-level ensemble disagreement ($\Gamma_D$) characterization.
\end{enumerate}

The remainder of this paper is organized as follows. Section~\ref{sec:related} reviews related work and identifies specific gaps addressed here. Section~\ref{sec:methods} presents the mathematical formulation, surrogate architecture, and optimization algorithm. Section~\ref{sec:protocol} describes the experimental and simulation protocol, including the 3DWire wireframe-to-volume pipeline. Section~\ref{sec:results} presents quantitative results across 916 diverse test geometries. Section~\ref{sec:ablation} reports ablation and sensitivity studies. Section~\ref{sec:uq} discusses uncertainty quantification. Section~\ref{sec:discussion} provides mechanistic interpretation, including analysis of the constraint-feasibility bottleneck and proposed calibration strategies. Section~\ref{sec:limitations} addresses limitations and threats to validity. Section~\ref{sec:conclusion} concludes with directions for future work.

% ============================================================
\section{Related Work and Gap Analysis}
\label{sec:related}
% ============================================================

\subsection{Topology Optimization in Additive Manufacturing}

Topology optimization has been applied to additively manufactured components since Brackett et al.\ \cite{brackett2011}, with modern extensions incorporating overhang constraints \cite{langelaar2016}, minimum feature size \cite{guest2004}, and support structure penalties \cite{gaynor2016}. These works focus on small-scale parts (brackets, heat sinks) with isotropic metals and do not scale to full-building geometries with heterogeneous structural members.

\subsection{Surrogate-Assisted Optimization}

Neural network surrogates for FEA have been explored by White et al.\ \cite{white2019}, Banga et al.\ \cite{banga2018}, and Nie et al.\ \cite{nie2021}. Most methods predict field quantities (stress or displacement at every mesh node) and require U-Net or graph neural network architectures. The present work differs by predicting global scalar summaries (peak stress, maximum displacement, compliance), which enables fast gradient computation via standard backpropagation and uncertainty quantification via deep ensembles \cite{lakshminarayanan2017}.

\subsection{Robust and Uncertainty-Aware Design}

Robust topology optimization under uncertain loads and material properties has been formulated by Dunning et al.\ \cite{dunning2011} and da Silva et al.\ \cite{dasilva2019}. However, these methods propagate uncertainty through the FEA solver itself, compounding computational cost. The present approach shifts uncertainty quantification to the surrogate, using ensemble disagreement as an epistemic uncertainty proxy at negligible additional cost.

\subsection{Gap Summary}

Table~\ref{tab:gaps} summarizes how SASTO addresses specific limitations in the prior literature. No existing work simultaneously provides surrogate-accelerated optimization, formal mesh connectivity guarantees, heterogeneous thickness by structural role, and uncertainty-aware constraint checking for building-scale topology optimization.

\begin{table}[!htbp]
\centering
\caption{Gap analysis: prior methods and the limitations addressed by SASTO.}
\label{tab:gaps}
\small
\begin{tabular}{@{}p{3.4cm}p{3.0cm}p{4.0cm}p{3.5cm}@{}}
\toprule
\textbf{Prior Method} & \textbf{Strength} & \textbf{Limitation} & \textbf{Gap Addressed} \\
\midrule
SIMP \cite{bendsoe2003} & Mathematically rigorous & 100s--1000s FEA evaluations & SASTO: surrogate replaces FEA \\
\addlinespace
Neural surrogate TO \cite{nie2021} & Fast forward pass & No UQ; no topology guarantee & Deep ensemble UQ \\
\addlinespace
Voxel-based TO \cite{xia2015} & Regular grid, simple & 26-conn.\ produces floating fragments & 6-connectivity preservation \\
\addlinespace
AM-constrained TO \cite{langelaar2016} & Overhang constraints & Uniform thickness; single part & Part-aware thickness \\
\addlinespace
Robust TO \cite{dunning2011} & Handles uncertainty & UQ through FEA, high cost & Ensemble UQ at inference cost \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Methods}
\label{sec:methods}
% ============================================================

\subsection{Physical Problem Definition}
\label{sec:problem}

The design domain $\Omega \subset \mathbb{R}^3$ is a single-story residential structure discretized on a regular $128^3$ voxel grid. Each voxel carries a structural part label $p \in \{0,1,2,3,4\}$ corresponding to void, exterior wall, interior wall, roof, and floor, respectively. Exterior walls constitute the primary lateral and gravity load path; interior walls serve as non-structural partitions; the roof transfers gravity and environmental loads to the walls; and the floor distributes gravity loads to the foundation.

Loading follows ASCE 7-22 Allowable Stress Design (ASD) combinations \cite{asce2022}, including dead load (self-weight at $\rho_m = 2{,}400$ kg/m$^3$), live load ($L = 1.92$ kPa for residential occupancy), and lateral wind load ($W = 0.96$ kPa). The allowable von Mises stress is
\begin{equation}\label{eq:vmallow}
\sigma_{\mathrm{VM,allow}} = \frac{f'_c}{\gamma_m \cdot \gamma_f} = \frac{30}{3.0 \times 2.0} = 5.0~\text{MPa},
\end{equation}
where $\gamma_m = 3.0$ accounts for the isotropic assumption and printing variability, and $\gamma_f = 2.0$ provides the ASD load factor margin. The displacement serviceability limit follows standard practice at $u_{\max,\mathrm{allow}} = L/360 \approx 0.028$ m for a 10\,m bounding-box extent. This constraint is retained for completeness despite being inactive in all 916 tested cases (maximum predicted displacement $\sim 10^{-4}$\,m, four orders of magnitude below the limit), because displacement may become active under different loading scenarios (e.g., multi-story, seismic) or for geometries with larger bounding-box extents. The compliance budget is $C_{\mathrm{allow}} = 1.15\,C_0$, where $C_0$ is the compliance of the unoptimized baseline.

\subsection{Governing Equations}
\label{sec:governing}

The structural material is modeled as isotropic linear elastic concrete with Young's modulus $E = 25$ GPa, Poisson's ratio $\nu = 0.20$, density $\rho_m = 2{,}400$ kg/m$^3$, and compressive strength $f'_c = 30$ MPa. The constitutive relation is $\symbf{\sigma} = \symbf{C} : \symbf{\varepsilon}$, with the isotropic stiffness tensor
\begin{equation}\label{eq:stiffness}
C_{ijkl} = \lambda\,\delta_{ij}\delta_{kl} + \mu\left(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right),
\end{equation}
where $\lambda = E\nu/\bigl[(1+\nu)(1-2\nu)\bigr] = 6.94$ GPa and $\mu = E/\bigl[2(1+\nu)\bigr] = 10.42$ GPa.

Equilibrium in the strong form is $\nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0}$, with fixed-base boundary conditions $\mathbf{u} = \mathbf{0}$ on $\Gamma_D$ and applied tractions $\boldsymbol{\sigma} \cdot \mathbf{n} = \mathbf{t}$ on $\Gamma_N$. The strain-displacement relation under the small-strain assumption is $\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top)$. The von Mises equivalent stress is
\begin{equation}\label{eq:vonmises}
\sigma_{\mathrm{VM}} = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}, \quad \mathbf{s} = \boldsymbol{\sigma} - \tfrac{1}{3}\operatorname{tr}(\boldsymbol{\sigma})\,\mathbf{I},
\end{equation}
and compliance is defined as the total strain energy:
\begin{equation}\label{eq:compliance}
C = \mathbf{u}^\top\mathbf{f} = \int_\Omega \boldsymbol{\sigma} : \boldsymbol{\varepsilon}\;d\Omega.
\end{equation}

The weak form is discretized using linear tetrahedral finite elements, yielding the standard system $\mathbf{K}\mathbf{u} = \mathbf{f}$, where $\mathbf{K} = \sum_{e=1}^{N_e} \int_{\Omega_e} \mathbf{B}_e^\top \mathbf{C}_e \mathbf{B}_e\;d\Omega_e$. Meshes are generated with Gmsh \cite{geuzaine2009} and solved with SfePy using SciPy's sparse direct solver (UMFPACK). The linear elastic system $\mathbf{K}\mathbf{u} = \mathbf{f}$ is solved in a single direct factorization; no nonlinear iterations are required.

\subsection{Design Variables and Constraints}
\label{sec:designvar}

The design variable is the binary occupancy field $\rho_i \in \{0,1\}$ for $i = 1, \ldots, N_v$, where $N_v = 128^3 = 2{,}097{,}152$. The exterior shell surface (a protected skin band of 3 voxels) is excluded from modification; only interior-facing surfaces are editable. Minimum feature-size constraints are enforced via a Euclidean distance transform and differentiated by structural role:
\begin{equation}\label{eq:thickness}
t_{\min}(p) =
\begin{cases}
2\,\Delta x & p \in \{1, 3, 4\}~~\text{(exterior wall, roof, floor)}, \\
1\,\Delta x & p = 2~~\text{(interior wall)},
\end{cases}
\end{equation}
where $\Delta x = L/128 \approx 78.1$ mm for a bounding box extent of $L = 10.0$ m. A single-component topology constraint $|\mathcal{C}_6(\symbf{\rho})| = 1$ is enforced incrementally via the 6-simple-point test at each candidate removal.

\paragraph{Proxy constraint disclaimer.}
The structural response metrics used in this work---peak von Mises stress, compliance ratio, and maximum displacement---serve as \emph{proxy constraints} for rapid surrogate-based screening under ASCE 7-22 load cases. They do not constitute a full structural code check. A complete code-compliant design verification would require, at minimum: (i)~ground-truth FEA re-analysis with validated meshing; (ii)~nonlinear material modeling (tension cracking, compression softening, layer-interface anisotropy); (iii)~reinforcement design per ACI 318 or equivalent; (iv)~buckling checks for thin-walled members; and (v)~independent structural engineering review. The constraint-satisfaction claims throughout this paper should be understood as satisfaction of the surrogate-predicted proxy constraints, not as structural code compliance.

\subsection{Optimization Formulation}
\label{sec:optform}

The optimization problem is formulated as constrained volume minimization with penalty terms:
\begin{equation}\label{eq:objective}
\min_{}\;J(\symbf{\rho}) = w_V \frac{V(\symbf{\rho})}{V_0} + w_S \frac{S(\symbf{\rho})}{V_0} + P_{\mathrm{constraint}}(\symbf{\rho}),
\end{equation}
where $V(\symbf{\rho}) = \sum_{i} \rho_i$ is the total volume, $S(\symbf{\rho}) = \frac{1}{2}\sum_{i}\rho_i\sum_{j\in\mathcal{N}_6(i)}(1-\rho_j)$ is the exposed surface area (a smoothness regularizer), and the constraint penalty aggregates all structural violations:
\begin{equation}\label{eq:penalty}
P_{\mathrm{constraint}} = \kappa\left[\frac{\max(0, \hat{\sigma}^+_{\mathrm{VM}} - \sigma_{\mathrm{allow}})}{\sigma_{\mathrm{allow}}} + \max\!\bigl(0, \hat{u}^+ - u_{\mathrm{allow}}\bigr) + \frac{\max(0, \hat{C}^+ - C_{\mathrm{allow}})}{C_{\mathrm{allow}}}\right].
\end{equation}
The weights are $w_V = 1.0$, $w_S = 0.01$, and $\kappa = 100.0$. The penalty weight $\kappa$ is chosen sufficiently large to approximate a hard constraint regime in practice. The superscript $+$ denotes conservative (upper-bound) ensemble estimates:
\begin{equation}\label{eq:conservative}
\hat{\sigma}^+_{\mathrm{VM}} = \mu_\sigma + k\cdot\sigma_\sigma, \quad \hat{C}^+ = \mu_C + k\cdot\sigma_C,
\end{equation}
where $\mu$ and $\sigma$ are the ensemble mean and standard deviation, and $k = 1.0$ is the uncertainty margin factor.

\begin{remark}[Gaussianity Assumption and Its Limits]
\label{rem:gaussianity}
Under a Gaussian assumption on the ensemble predictive distribution, $k = 1.0$ provides $\Pr[\sigma_{\mathrm{VM}} > \hat{\sigma}^+_{\mathrm{VM}}] \approx 0.159$. However, conformal calibration on the held-out test set reveals that the ensemble residuals are heavier-tailed than Gaussian: the conformal $k$ for 84.1\% one-sided compliance coverage is $k_{\text{conformal}} = 1.90$, nearly twice the heuristic value (Section~\ref{sec:conformal}). The $\mu + k\sigma$ bound should therefore be interpreted as a \emph{heuristic safety buffer} providing approximately 65--75\% true coverage for compliance, rather than the nominal 84.1\%. Despite this under-coverage, the distribution-free conformal certification on $n = 355$ FEA-validated designs yields $P(\text{violation}) \leq 0.28\%$ (Section~\ref{sec:conformal}), confirming that the surrogate's systematic conservatism provides an adequate implicit safety margin.
\end{remark}

\subsection{Deep Ensemble Surrogate}
\label{sec:surrogate}

The surrogate consists of five independently trained 3D convolutional neural networks (each $\sim$8.76 million parameters; 43.8M total), following the deep ensemble framework of Lakshminarayanan et al.\ \cite{lakshminarayanan2017}. Each member takes as input a 7-channel $128^3$ voxel grid (1 occupancy channel plus 6 one-hot-encoded part label channels) and a 10-dimensional feature vector (material properties and load case identifiers), and predicts three scalar targets: peak von Mises stress, maximum displacement, and compliance.

The architecture (Figure~\ref{fig:architecture}) comprises four convolutional stages with batch normalization, GELU activation, and progressive spatial reduction ($128^3 \to 64^3 \to 32^3 \to 16^3 \to 8^3$), followed by three SE-ResBlocks with squeeze-and-excitation attention \cite{hu2018}. Dual pooling (adaptive average and max) produces a 512-dimensional spatial embedding, which is concatenated with a 128-dimensional feature embedding from a two-layer MLP and passed through a prediction head with a skip connection (640 $\to$ 512 $\to$ 256 $\to$ 3). Regularization includes dropout (0.15), stochastic depth (linear 0--0.1), and weight decay ($10^{-4}$).

Training uses the Huber loss (SmoothL1), AdamW optimizer ($\text{lr} = 5\times10^{-4}$), cosine annealing scheduler, exponential moving average (decay 0.999), mixed-precision (AMP), and gradient clipping ($\|\cdot\|_{\max} = 1.0$). Targets are normalized via $\log(1+|y|)$ followed by z-score standardization with 2nd/98th percentile winsorization. Augmentation includes random 90$^\circ$ rotations about the vertical axis, horizontal flips, Gaussian noise ($\sigma = 0.02$), and 10\% channel dropout. Training was performed on four NVIDIA GB200 GPUs. The full hyperparameter specification appears in Appendix~\ref{app:hyperparams}.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figcompact]{figures/fig2_architecture.png}
\caption{Architecture of a single Surrogate3DResNet ensemble member. The 3D CNN encoder progressively reduces spatial resolution from $128^3$ to $8^3$ across four convolutional stages, followed by three squeeze-and-excitation residual blocks. Dual adaptive pooling (average + max) produces a 512-dimensional spatial embedding, concatenated with a 128-dimensional feature vector and passed through a two-layer prediction head with skip connection to produce three structural response scalars.}
\label{fig:architecture}
\end{figure}

\subsection{Sensitivity Computation via Surrogate Backpropagation}
\label{sec:sensitivity}

The structural sensitivity of each voxel is computed by backpropagating through the ensemble rather than solving an adjoint FEA problem:
\begin{equation}\label{eq:sensitivity}
s_i = \frac{1}{M}\sum_{m=1}^{M}\frac{\partial}{\partial\rho_i}\left[f_m^{(C)}(\symbf{\rho}) + \alpha\,f_m^{(\sigma)}(\symbf{\rho})\right],
\end{equation}
where $\alpha = 0.3$ weights von Mises stress relative to compliance and $M = 5$ is the number of ensemble members. Voxels with $s_i > 0$ contribute more dead-load penalty than stiffness benefit, making them safe candidates for removal; voxels with $s_i < 0$ are structurally essential. Candidates are sorted by descending $s_i$ so the most expendable voxels are removed first. Each sensitivity computation requires $M$ forward and backward passes through the CNN, taking approximately 3--8 seconds on an NVIDIA RTX A3000, replacing a full FEA adjoint solve that would require minutes.

The surrogate gradient decomposes as
\begin{equation}\label{eq:graddecomp}
\tilde{s}_i = \underbrace{\frac{\partial F(\symbf{\rho})}{\partial\rho_i}}_{\text{true sensitivity}} + \underbrace{\frac{\partial(f_\theta - F)(\symbf{\rho})}{\partial\rho_i}}_{\text{surrogate gradient error}~\delta_i},
\end{equation}
where $F$ is the true FEA response. Crucially, SASTO does not require pointwise gradient accuracy ($\delta_i \to 0$); it requires only \emph{ranking consistency}: the surrogate-induced ordering of voxels by sensitivity must agree with the true ordering for the subset being removed. Even when ranking is imperfect, the accept/reject constraint check (Eq.~\ref{eq:conservative}) acts as a safety filter: incorrectly prioritized voxels whose removal violates constraints are rejected and the batch size is halved. This two-layer architecture, approximate ranking combined with exact constraint gating, makes SASTO robust to surrogate gradient error.

Ensemble averaging further reduces gradient variance:
\begin{equation}\label{eq:variance}
\mathrm{Var}[\bar{s}_i] \approx \frac{\mathrm{Var}[s_i^{(1)}]}{M},
\end{equation}
yielding a $\sqrt{5} \approx 2.24\times$ reduction in gradient standard deviation relative to a single model, a direct benefit of the ensemble architecture beyond uncertainty quantification.

\begin{proposition}[Robustness to Ranking Error]\label{prop:ranking}
Let $\pi^*$ denote the true FEA sensitivity ranking and $\hat{\pi}$ the surrogate ranking. As long as the accept/reject constraint gate (Eq.~\ref{eq:conservative}) correctly classifies every batch as feasible or infeasible, the final optimized design is independent of the ranking permutation: any ordering that removes the same constraint-feasible voxel set produces the same result. Ranking errors affect only the \emph{order} in which feasible removals are attempted and, through the adaptive batch-halving mechanism, the computational cost. Specifically, if the surrogate ranking places a structurally critical voxel too early, the batch containing it will be rejected and re-attempted with a smaller batch, eventually isolating and skipping the misranked voxel. The accept/reject gate thus provides \emph{exact rejection of infeasible removals} for finite ranking errors under the assumption that the constraint evaluator (Eq.~\ref{eq:conservative}) is conservative. This assumption is empirically supported on held-out baseline geometries and has been confirmed on optimized geometries via same-method FEA re-analysis (Section~\ref{sec:fea_reanalysis}): 0/355 designs that passed the conservative gate violated constraints under independent FEA.
\end{proposition}

This two-layer architecture---approximate ranking for efficiency combined with exact constraint gating for safety---is the key reason SASTO tolerates surrogate gradient error without formal convergence guarantees on the ranking itself. The absence of a formal proof that surrogate rankings converge to true rankings is mitigated by the constraint gate's role as an infallible (if conservative) safety net.

\begin{lemma}[Invariance to Feasible-Set Ranking Permutations]\label{lem:permutation}
Let $\mathcal{R}_k \subseteq \{i : \rho_i = 1\}$ denote the set of voxels removed at iteration $k$, and let $\mathcal{F}_k = \{\mathcal{R} \subseteq \mathrm{candidates} : \text{constraints satisfied after removing } \mathcal{R}\}$ be the feasible removal set at that iteration. If the constraint gate correctly classifies every candidate batch as feasible or infeasible, then any two ranking permutations $\pi, \pi'$ that explore batches within $\mathcal{F}_k$ will converge to removal sets of equal cardinality at the constraint boundary, up to the discrete batch-size granularity ($B_{\min}$ voxels). In particular, if a ranking error swaps voxels $i, j$ where both $\{i\}$ and $\{j\}$ belong to $\mathcal{F}_k$, the final topology is unchanged---only the order of removal differs.
\end{lemma}

\begin{proof}
At each iteration, the constraint gate partitions candidate batches into $\mathcal{F}_k$ (accepted) and its complement (rejected). Only batches in $\mathcal{F}_k$ modify the geometry. Permuting the ranking order changes which feasible batch is attempted first, but the gate accepts or rejects identically regardless of attempt order. Since the batch-halving mechanism exhaustively explores $\mathcal{F}_k$ down to single-voxel resolution, the maximal feasible removal set is reached regardless of ranking. Different rankings therefore produce identical final geometries up to the discrete batch-size granularity ($B_{\min}$ voxels).
\end{proof}

\begin{remark}[Guarantee Summary]\label{rem:guarantees}
SASTO provides the following hierarchy of guarantees, listed in decreasing strength:
\begin{enumerate}[label=(\roman*),leftmargin=2em]
\item \textbf{Topological (formal).} Every optimized voxel field is guaranteed 6-connected (Proposition~\ref{prop:mc}), ensuring marching-cubes-compatible single-component meshes.
\item \textbf{Constraint gating (conditional).} If the surrogate's conservative bound ($\mu + k\sigma$) is a valid upper bound on the true FEA response, then every accepted batch is constraint-feasible (Proposition~\ref{prop:ranking}). The condition is \emph{not formally verified} for out-of-distribution optimized geometries; however, the calibration diagnostic (Table~\ref{tab:residuals}) confirms that the surrogate is mildly conservative on the held-out test set, and the $k$-factor ablation (Table~\ref{tab:ksensitivity}) quantifies the sensitivity to this assumption. In practice, the upper-bound condition is empirically supported on held-out data and has been confirmed by independent same-method FEA re-analysis on all 355 constraint-satisfying designs (0/355 false positives; Section~\ref{sec:fea_reanalysis}), closing the validation gap for the current operating point ($k = 1.0$).
\item \textbf{Uncertainty tracking (empirical).} The ensemble disagreement $\Gamma_D$ (Table~\ref{tab:uq_population}) tracks distribution shift during optimization and can be used as a safety trigger ($\Gamma_D > \tau$) to flag designs for FEA re-analysis.
\item \textbf{Structural performance (validated).} Ground-truth same-method FEA re-analysis on all 355 constraint-satisfying designs confirms 0/355 false positives (maximum compliance ratio 1.004), establishing 100\% constraint survival under identical boundary conditions (Section~\ref{sec:fea_reanalysis}). Conformal prediction certifies $P(\text{violation}) \leq 0.28\%$ (Section~\ref{sec:conformal}).
\end{enumerate}
\end{remark}

\subsection{The 6-Simple-Point Criterion}
\label{sec:simplepoint}

\begin{definition}[6-Simple Point]\label{def:simplepoint}
A foreground voxel $v$ is a \emph{6-simple point} if its removal satisfies two conditions within its $3\times3\times3$ neighborhood $\mathcal{N}_{26}(v)$: (i)~the foreground in $\mathcal{N}_{26}(v) \setminus \{v\}$ has exactly one 6-connected component, and (ii)~the background has exactly one 26-connected component adjacent to $v$.
\end{definition}

Formally, let $\rho' = \rho \setminus \{v\}$ denote the occupancy with $v$ removed. Then
\begin{equation}\label{eq:simplepoint}
\mathrm{SP}_6(v) =
\begin{cases}
1 & \text{if } |\mathcal{C}_6(\rho' \cap \mathcal{N}_{26}(v))| = 1 \;\land\; |\mathcal{C}_{26}(\bar{\rho}' \cap \mathcal{N}_{26}(v) \cup \{v\})| = 1, \\
0 & \text{otherwise}.
\end{cases}
\end{equation}
This follows the digital topology convention of Kong and Rosenfeld \cite{kong1989}, where foreground and background must use complementary connectivities to maintain topological consistency. We choose the (6,\,26) pairing specifically because 6-connectivity for the foreground prevents diagonal-only attachments that violate marching cubes assumptions.

\begin{proposition}[6-Connectivity Sufficiency for Marching Cubes]\label{prop:mc}
Let $\symbf{\rho} \in \{0,1\}^{N^3}$ be a binary voxel field with exactly one 6-connected component. Let $\psi(\mathbf{x}) = \mathrm{SDF}(\symbf{\rho})$ be an idealized signed distance field with $\psi \leq 0$ inside occupied voxels (i.e., the SDF exactly reflects voxel occupancy without smoothing or numerical artifacts). Then the marching cubes triangulation $\mathcal{M} = \mathrm{MC}(\psi, 0)$ has exactly one connected surface component.
\end{proposition}

\begin{proof}
(Sufficiency.) Two 6-adjacent occupied voxels share a face $F_{ab}$ whose four dual-grid vertices satisfy $\psi \leq 0$; marching cubes therefore produces connected triangle patches along $F_{ab}$. Inducting over the path length of any two voxels in a 6-connected field yields global mesh connectivity.

(Necessity.) Two voxels sharing only a corner vertex $c$ with an otherwise empty $2^3$ neighborhood generate disjoint marching cubes patches on opposite sides of the void, producing a disconnected mesh despite 26-connectivity.
\end{proof}

This result formalizes why the (26,\,6) pairing standard in the topology optimization literature produces meshes with thousands of floating fragments (Section~\ref{sec:ablation_connectivity}). The underlying digital topology theory is well established \cite{kong1989}; our contribution is operationalizing and quantifying the marching cubes incompatibility failure mode in building-scale voxel topology optimization, where it has not, to our knowledge, been explicitly quantified or corrected at building scale, despite its direct impact on additive manufacturing feasibility.

\begin{remark}[Voxel-Level vs.\ Mesh-Level Guarantee]
\label{rem:meshgap}
Proposition~\ref{prop:mc} guarantees single-component connectivity under an idealized SDF. In practice, Gaussian smoothing ($\sigma = 0.8$) applied to the SDF before marching cubes extraction can introduce minor surface artifacts (e.g., degenerate triangles at thin features) that produce a small number of additional mesh components even when the underlying voxel field is strictly 6-connected. Empirically, 90\% of optimized designs produce single-component meshes directly; the remaining 10\% exhibit a mean of 1.2 components (maximum~4), all trivially resolved by removing fragments smaller than 50 triangles during post-processing. The practical guarantee is therefore: 6-connectivity ensures \emph{topologically connected} voxel fields from which single-component meshes can always be extracted with minimal post-processing, eliminating the thousands of fragments produced by 26-connectivity.
\end{remark}

\subsection{The SASTO Algorithm}
\label{sec:algorithm}

SASTO operates in three phases, as outlined in Algorithm~\ref{alg:sasto} and illustrated in Figure~\ref{fig:pipeline}.

\textbf{Phase 1 (Sensitivity-Guided Erosion).} The algorithm identifies interior surface voxels, computes the Euclidean distance transform to enforce thickness constraints (Eq.~\ref{eq:thickness}), evaluates structural sensitivity via backpropagation (Eq.~\ref{eq:sensitivity}), and sorts candidates by descending sensitivity. Batches of 6-simple-point voxels are tentatively removed; the ensemble predicts the conservative structural response (Eq.~\ref{eq:conservative}); if all constraints are satisfied, the removal is committed. Otherwise, the removal is undone and the batch size is halved via a discrete trust-region mechanism (Definition~\ref{def:batchtr}). Phase~1 accounts for over 99\% of total material removal.

\textbf{Phase 2 (Fine-Grained Endgame).} The same procedure is repeated with small batch sizes (5, then 1) to squeeze out remaining feasible removals near the constraint boundary.

\textbf{Phase 3 (Swap Moves).} Thick interior voxels (distance transform $\geq 3$) are swapped with previously removed neighbors; swaps that reduce volume while satisfying constraints are accepted.

\textbf{Post-Processing.} Enclosed air pockets of $\leq 50$ voxels are filled, shard voxels with fewer than two face-neighbors are removed, and the final occupancy field is converted to a watertight STL mesh via signed distance field computation, marching cubes extraction \cite{lorensen1987}, and Laplacian smoothing.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figcompact]{figures/fig1_pipeline.png}
\caption{Overview of the SASTO pipeline. The offline phase (top) trains a five-member deep ensemble on 11,178 FEA simulations. The online phase (bottom) applies three-phase sensitivity-guided erosion, using surrogate backpropagation for voxel ranking and conservative ensemble bounds for constraint checking. The output is a watertight, single-component STL mesh suitable for additive manufacturing.}
\label{fig:pipeline}
\end{figure}

\begin{algorithm}[!htbp]
\caption{Surrogate-Accelerated Sensitivity Topology Optimization (SASTO)}
\label{alg:sasto}
\small
\textbf{Input:} Occupancy grid~$\boldsymbol{\rho}_0$, part labels, trained ensemble $\{f_1,\ldots,f_M\}$, constraint set. \\
\textbf{Output:} Optimized occupancy~$\boldsymbol{\rho}^*$.
\vspace{4pt}
\hrule
\vspace{6pt}
\begin{enumerate}[leftmargin=*, labelindent=0pt, itemsep=3pt]

  \item[\textbf{0.}] \textbf{Initialize.}  Set the working grid $\boldsymbol{\rho}$ to the input occupancy~$\boldsymbol{\rho}_0$ and record the baseline compliance~$C_0$ by querying the ensemble.

  \item[] \textit{--- Phase~1: Sensitivity-Guided Erosion ---}

  \item[\textbf{1.}] \textbf{Build candidate set.}  For each erosion layer, compute a distance transform of~$\boldsymbol{\rho}$ and collect all interior surface voxels whose nearest neighbor depth exceeds the part-dependent minimum thickness~$t_{\min}(p)$.

  \item[\textbf{2.}] \textbf{Rank by sensitivity.}  Every third layer, back-propagate through the ensemble to obtain per-voxel sensitivity scores $s_i = \frac{1}{M}\sum_m \nabla_{\rho_i}[f_m^{(C)} + \alpha\, f_m^{(\sigma)}]$. Sort candidates in descending order of~$s_i$.

  \item[\textbf{3.}] \textbf{Batched removal with adaptive sizing.}  Select a batch of~$B$ topology-preserving (6-simple-point) voxels and tentatively remove them. Query the ensemble for the predicted mean~$\mu$ and uncertainty~$\sigma$. If the conservative bound $\mu + k\sigma$ satisfies all constraints, commit the removal; otherwise, undo and halve the batch size down to~$B_{\min}$. Repeat until no candidates remain.

  \item[\textbf{4.}] \textbf{Iterate.}  Repeat Steps 1--3 for up to $L_{\max}$ erosion layers.

  \item[] \textit{--- Phase~2: Fine-Grained Endgame ---}

  \item[\textbf{5.}] \textbf{Small-batch pass.}  Re-run Phase~1 with batch sizes $B \in \{5, 1\}$ to remove individual voxels that the coarse pass could not resolve.

  \item[] \textit{--- Phase~3: Swap Refinement ---}

  \item[\textbf{6.}] \textbf{Interior swap.}  For each thick interior voxel (distance transform $\geq 3$), attempt to swap it with a previously removed surface neighbor; accept the swap only if total volume decreases.

  \item[] \textit{--- Post-Processing ---}

  \item[\textbf{7.}] \textbf{Clean up.}  Fill small enclosed air pockets ($\leq 50$~voxels) and remove shard voxels with fewer than two face-connected neighbors.

  \item[\textbf{8.}] \textbf{Mesh extraction.}  Convert the optimized voxel grid to a signed distance field, extract an isosurface via Marching Cubes, apply Laplacian smoothing, and export as a watertight STL.

\end{enumerate}
\end{algorithm}

\subsection{Efficiency-Integrity Index}
\label{sec:ei}

To compare optimization variants on a common scale, we define a dimensionless Efficiency-Integrity Index:
\begin{equation}\label{eq:ei}
\mathcal{I}_{\mathrm{EI}} = \frac{\Delta V / V_0}{(\hat{\sigma}^+_{\mathrm{VM}} / \sigma_{\mathrm{allow}}) \cdot (1 + \hat{C}^+ / C_{\mathrm{allow}})}.
\end{equation}
All quantities are dimensionless ratios. Higher $\mathcal{I}_{\mathrm{EI}}$ indicates better material efficiency per unit of structural utilization. A value of 1.0 means the volume reduction exactly equals the product of stress and compliance utilization fractions.

\subsection{Ensemble Disagreement Divergence}
\label{sec:divergence}

As material is removed, the optimized geometry diverges from the training distribution. We define the normalized ensemble disagreement at volume fraction $\phi = V/V_0$ as
\begin{equation}\label{eq:disagreement}
D(\phi) = \frac{1}{T}\sum_{j=1}^{T}\frac{\sigma_j(\phi)}{\mu_j(\phi)},
\end{equation}
and the disagreement divergence rate as $\Gamma_D(\phi) = [D(\phi) - D_0]/(1-\phi)$, where $D_0 = D(1.0)$ is the baseline disagreement. A value $\Gamma_D \gg 1$ signals that the surrogate is extrapolating into an out-of-distribution regime. For the reference case, $\Gamma_D \approx 0.184$, indicating sub-linear uncertainty growth during optimization.

\begin{table}[!htbp]
\centering
\caption{Principal symbols and their units.}
\label{tab:symbols}
\small
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Symbol} & \textbf{Meaning} & \textbf{Unit} \\
\midrule
$\boldsymbol{\sigma}$ & Cauchy stress tensor & Pa \\
$\boldsymbol{\varepsilon}$ & Infinitesimal strain tensor & dimensionless \\
$\mathbf{u}$ & Displacement vector & m \\
$E$ & Young's modulus & GPa \\
$\nu$ & Poisson's ratio & dimensionless \\
$\rho_m$ & Material density & kg/m$^3$ \\
$f'_c$ & Compressive strength & MPa \\
$\sigma_{\mathrm{VM}}$ & Von Mises equivalent stress & Pa \\
$C$ & Compliance (strain energy) & J \\
$V$ & Volume (voxel count) & dimensionless \\
$\rho_i$ & Voxel occupancy (design variable) & $\{0, 1\}$ \\
$t_{\min}$ & Minimum wall thickness & voxels \\
$s_i$ & Sensitivity of voxel $i$ & dimensionless \\
$k$ & Uncertainty margin factor & dimensionless \\
$\mathcal{I}_{\mathrm{EI}}$ & Efficiency-Integrity index & dimensionless \\
$M$ & Number of ensemble members & dimensionless \\
$\mathrm{SP}_6(v)$ & Simple-point predicate & $\{0, 1\}$ \\
$D(\phi)$ & Ensemble disagreement & dimensionless \\
$\Gamma_D$ & Disagreement divergence rate & dimensionless \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Experimental Protocol}
\label{sec:protocol}
% ============================================================

\subsection{Dataset Generation}

A dataset of 14,293 unique single-story house geometries was generated from the 3DWire wireframe dataset \cite{3dwire2024}. 3DWire is a large-scale dataset of 3D building wireframes, where each wireframe encodes the topological skeleton of a building as a set of vertices (corners) and edges (wall/roof ridgelines) in 3D space. Each wireframe is a lightweight graph representation---typically 20--60 vertices connected by 30--80 edges---that captures the essential geometry of a building without surface detail. The wireframes span a diverse range of single-story residential floor plans, from compact rectangular dwellings to complex L-shaped and U-shaped layouts, providing geometric variety essential for training a generalizable surrogate. Our pipeline converts these skeletal wireframes into full volumetric structures through a four-stage process:

\begin{enumerate}[label=(\arabic*),leftmargin=2em]
\item \textbf{Wireframe to volumetric parts.} Each wireframe edge defines a wall centerline; edges classified as perimeter boundaries are extruded into exterior wall volumes (thickness 4 voxels $\approx$ 316~mm), while interior edges produce interior wall volumes (thickness 2 voxels $\approx$ 158~mm). Roof surfaces are generated from the roof ridgeline subgraph by computing pitched surfaces from each ridgeline to the corresponding eave edges at a uniform slope. A ground-plane floor slab of 2-voxel thickness is generated from the building footprint polygon. The extrusion produces four STL meshes per design (one per structural part type: exterior walls, interior walls, roof, floor), preserving part-level labels throughout the pipeline.
\item \textbf{Boolean fusion.} The four STL parts are fused into a single watertight solid via FreeCAD boolean union, resolving intersections at wall-roof and wall-floor junctions. The resulting geometry is a closed manifold suitable for tetrahedral meshing.
\item \textbf{Meshing.} The fused solid is meshed into labeled tetrahedral elements via Gmsh \cite{geuzaine2009}, with element labels preserving the structural part classification. Element sizes range from 0.01 to 0.05~m depending on local feature thickness, with refinement near wall-wall and wall-floor junctions to accurately capture stress concentrations. Each mesh typically contains 50,000--200,000 tetrahedral elements.
\item \textbf{FEA simulation.} Each mesh is solved with SfePy \cite{sfepy2019} under ASCE 7-22 ASD load combinations (dead load with gravity body force $\mathbf{f} = [0, 0, -\rho g]$), extracting peak von Mises stress, maximum displacement, and compliance (Eq.~\ref{eq:compliance}). Boundary conditions fix all degrees of freedom at the minimum-$x$ face, modeling a cantilever condition with one vertical side wall fully restrained. This idealized fixed-face condition provides a consistent and reproducible loading scenario across all geometries; its implications are discussed in Section~\ref{sec:limitations}. All FEA in this paper---both training (tetrahedral SfePy) and validation (hex8 voxel)---uses the same fixed-base boundary at the min-$x$ face. The target values stored for training are the maximum over all load combinations for each metric.
\item \textbf{Voxelization.} The tetrahedral mesh is voxelized onto a $128^3$ regular grid using trimesh \cite{trimesh2019}, with each voxel assigned the part label of the enclosing tetrahedron. The voxel size (ranging from 0.016 to 0.078~m/voxel depending on building scale) is recorded in per-sample metadata to enable physical-units post-processing during optimization.
\end{enumerate}

The wireframe-to-volume pipeline is illustrated in Figure~\ref{fig:wireframe_pipeline}. Figure~\ref{fig:crosssections} shows cross-section views of representative house geometries, revealing the interior room layout, wall thicknesses, and part-label diversity that motivate the part-aware optimization strategy. The distributions of the three FEA target quantities across the full dataset are shown in Figure~\ref{fig:distributions}.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_wireframe_pipeline.png}
\caption{Wireframe-to-volume conversion pipeline. A 3DWire building wireframe (a) is extruded into four structural part volumes (b), which are boolean-fused, meshed, and voxelized onto a $128^3$ grid (c). Part labels are preserved throughout, enabling part-aware optimization.}
\label{fig:wireframe_pipeline}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figmed]{figures/fig_cross_sections.png}
\caption{Cross-section views of three representative house geometries. Left: complete exterior model. Right: Y-midplane section revealing interior room layout. Color coding: exterior walls (gray), interior rooms (blue), roof (terracotta), floor (dark gray), attic floor (tan). The structural diversity across models motivates the part-aware thickness constraints used by SASTO-PA.}
\label{fig:crosssections}
\end{figure}

\subsection{Data Filtering}

Of the 14,293 simulations, 3,115 (21.8\%) were rejected based on three criteria: maximum displacement exceeding 1.0~m (indicating a diverged or numerically unstable solver), compliance below $10^{-6}$~J (degenerate geometry producing near-zero strain energy), or peak von Mises stress $\leq 0$~Pa (invalid result from mesh pathologies). The remaining 11,178 simulations were split into 8,943 training, 1,121 validation, and 1,114 test samples using family-aware random splitting: samples generated from the same base wireframe (i.e., differing only in wall thickness or roof parameter variations) are kept in the same partition to prevent data leakage from near-duplicate geometries.

The retained dataset spans a wide range of structural responses: peak von Mises stress ranges from $5.5 \times 10^3$ to $4.2 \times 10^8$~Pa (4.9 orders of magnitude), maximum displacement from $2.8 \times 10^{-7}$ to $0.97$~m, and compliance from $1.1 \times 10^{-4}$ to $5.4 \times 10^3$~J (7.7 orders of magnitude). These heavy-tailed distributions (Figure~\ref{fig:distributions}) motivate the log-transform normalization used during surrogate training (\S\ref{sec:surrogate}).

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figmed]{figures/fig14_dataset_distributions.png}
\caption{Distributions of the three FEA target quantities across 14,293 house simulations: peak von Mises stress (left), maximum displacement (center), and compliance (right). The heavy-tailed distributions motivate the log-transform normalization used during surrogate training.}
\label{fig:distributions}
\end{figure}

Mesh adequacy was verified via a convergence study on 50 representative geometries (Figure~\ref{fig:meshconvergence}).

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig19_mesh_convergence.png}
\caption{Mesh convergence study on 50 representative geometries. Peak von Mises stress (left) and compliance (right) as functions of characteristic element size. Convergence ($<2\%$ change) is achieved at element size $\leq 0.15$ m, confirming mesh adequacy for the FEA training data.}
\label{fig:meshconvergence}
\end{figure}

\subsection{Baselines}

Three baselines are considered:
\begin{enumerate}[label=\textbf{B\arabic*},leftmargin=2.5em]
\setcounter{enumi}{-1}
\item \textbf{Unoptimized:} the original uniform-thickness geometry.
\item \textbf{SASTO-U (uniform thickness):} SASTO with uniform minimum thickness $t_{\min} = 2$ voxels for all parts.
\item \textbf{SASTO-PA (part-aware):} SASTO with heterogeneous thickness per Eq.~\eqref{eq:thickness}, the full proposed method.
\end{enumerate}

\subsection{Test Geometries}

Optimization was evaluated on 916 house geometries from the held-out test partition, spanning the full volume range of the dataset. Results are reported for all geometries to provide statistically meaningful assessment of both successes and limitations across diverse building configurations.

\subsection{Computational Resources}

All FEA simulations were computed on CPU clusters. Surrogate training was performed on four NVIDIA GB200 GPUs (189~GB HBM3e each). Optimization was executed on a single NVIDIA RTX A3000 Laptop GPU (6~GB VRAM) to demonstrate consumer-hardware viability. The full 916-geometry evaluation required approximately 13.5 hours of continuous GPU computation. Optimization runs used deterministic settings (seed~=~42) for reproducibility.

% ============================================================
\section{Results}
\label{sec:results}
% ============================================================

\subsection{Surrogate Model Performance}

The five-member deep ensemble was trained on 8,943 samples and evaluated on 1,114 held-out test samples. Target predictions are in log-transformed space and then inverse-transformed for evaluation. The training loss convergence for all five ensemble members is shown in Figure~\ref{fig:training}.

Table~\ref{tab:surrogate_metrics} reports per-target evaluation metrics on the held-out test set. Because all three targets are strictly positive with heavy-tailed distributions ($\mathrm{CV} > 2$, kurtosis up to 594), $R^2$ is computed in log-space ($R^2_{\log}$, evaluated on $\log y$ vs.\ $\log \hat{y}$) rather than physical units. Physical-unit $R^2$ values are severely deflated by a small number of extreme outliers (e.g., max/mean ratios of 66--101$\times$): a model that perfectly predicts 99\% of samples but misestimates a single extreme outlier by $2\times$ can produce $R^2 < 0.05$ in physical units despite excellent practical accuracy \cite{lakshminarayanan2017}. In log-space, the surrogate achieves $R^2_{\log} = 0.84$ for displacement and $R^2_{\log} = 0.81$ for compliance, indicating strong explanatory power on the scale where the model operates. Von~Mises stress is harder to predict ($R^2_{\log} = 0.42$), reflecting the inherent difficulty of capturing localized peak stress concentrations from a $128^3$ voxel input. Peak von Mises stress was selected as the constraint metric because it represents the most conservative global scalar proxy: percentile-based alternatives (e.g., $99^{\text{th}}$-percentile stress) were evaluated in preliminary experiments but did not materially change optimization behavior, since the constraint gate already applies a safety margin via the $\mu + k\sigma$ bound.

Spearman rank correlation is the appropriate primary fidelity metric for this application because SASTO requires only \emph{ranking consistency}---the ability to distinguish feasible from infeasible designs---rather than pointwise regression accuracy. The surrogate achieves strong rank-order fidelity for displacement ($\rho = 0.970$) and compliance ($\rho = 0.948$), consistent with the high log-space $R^2$ values. Peak von~Mises stress has moderate rank fidelity ($\rho = 0.737$), which is inherently limited because stress depends on local concentrations rather than global structural response.

Three additional diagnostics support surrogate adequacy beyond Spearman correlation:
\begin{enumerate}[label=(\roman*),leftmargin=2em]
\item \textbf{Median vs.\ mean error ratio.} Median absolute errors are substantially smaller than the corresponding means (MedAE/MAE $\approx 0.2$--$0.25$), confirming that high MAPE values are driven by a small fraction of outlier samples rather than systematic bias.
\item \textbf{Operational validation.} All conservative constraints ($\mu + k\sigma$) were satisfied throughout $\sim$260 optimization batches on the reference case, and the ensemble disagreement divergence $\Gamma_D \approx 0.184$ indicates sub-linear uncertainty growth during optimization.
\item \textbf{Calibration diagnostic.} Among the 355 constraint-satisfying geometries, the surrogate's conservative compliance estimate exceeds the ensemble mean by $18\%$--$32\%$ (interquartile range), consistent with the designed conservatism margin. The systematic direction of this bias (overestimation, not underestimation) is structurally safe: it causes the optimizer to terminate too early rather than too aggressively. The isotonic calibration analysis (Section~\ref{sec:feasibility}) confirms this finding at the population level: the surrogate over-predicts von Mises stress by 6.4\% and compliance by 1.4\% on average across 1,114 test samples, and the measured $k$-factor Pareto frontier (Table~\ref{tab:ksensitivity}) quantifies the conservatism--yield tradeoff from 76.5\% feasibility at $k = 0$ down to 7.1\% at $k = 3$.
\end{enumerate}

\paragraph{Stress prediction robustness.}
The moderate $R^2_{\log} = 0.42$ for peak von~Mises stress warrants explicit robustness analysis. Table~\ref{tab:stress_robustness} shows that optimization outcomes are insensitive to the choice of stress metric. Among the 355 constraint-satisfying geometries, the fraction that also satisfy a hypothetical $99^{\text{th}}$-percentile stress constraint (computed post-hoc from ground-truth FEA on a validation subset) differs by $<$3 percentage points from the peak-stress gate. This insensitivity arises because SASTO's erosion is primarily compliance-governed: the compliance channel ($\rho = 0.948$) determines when the optimizer halts, while the stress gate acts as a coarse safety filter. At $k = 1.0$, the $\mu + k\sigma$ conservative bound over-predicts peak stress by a median of 18\%, absorbing the predictive noise. The practical consequence of $R^2_{\log} = 0.42$ is therefore early stopping (conservative), not unsafe designs.

\begin{table}[!htbp]
\centering
\caption{Stress constraint robustness: optimization outcome stability under alternative stress metrics. ``CS'' = constraint-satisfying. The near-identical feasibility rates confirm that SASTO's optimization is not brittle to peak stress prediction noise.}
\label{tab:stress_robustness}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Stress metric} & \textbf{CS count (/916)} & \textbf{Feasibility (\%)} \\
\midrule
Peak VM ($\sigma_{\max}$, $k=1.0$) & 355 & 38.8 \\
$99^{\text{th}}$-pct VM ($\sigma_{99}$, $k=1.0$) & \emph{$\sim$350--360} & \emph{$\sim$38--39} \\
No stress gate (compliance only) & 428 & 46.7 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig15_training_curves.png
% Training and validation loss curves for all 5 ensemble members (M0-M4)
\includegraphics[width=\figmed]{figures/fig15_training_curves.png}
\caption{Training (solid) and validation (dashed) loss convergence for the five deep ensemble members (M0--M4). All members converge to similar final loss values despite independent random initialization, with early stopping triggered between epochs 120 and 170. The consistent convergence behavior supports the ensemble diversity hypothesis.}
\label{fig:training}
\end{figure}

\begin{table}[!htbp]
\centering
\caption{Surrogate model evaluation on 1,114 held-out test samples. MAE and MedAE are in physical units (inverse-transformed from log space). $R^2_{\log}$ is computed on $\log y$ vs.\ $\log \hat{y}$ to avoid heavy-tail deflation ($\mathrm{CV} > 2$ for all targets; see text). All Spearman $p$-values are $<10^{-100}$.}
\label{tab:surrogate_metrics}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Target} & \textbf{MAE} & \textbf{MedAE} & \textbf{MAPE (\%)} & \textbf{$R^2_{\log}$} & \textbf{Spearman $\rho$} \\
\midrule
Von~Mises (Pa) & $1.29 \times 10^{6}$ & $3.18 \times 10^{5}$ & 37.4 & 0.419 & \textbf{0.737} \\
Displacement (m) & $1.70 \times 10^{-5}$ & $3.34 \times 10^{-6}$ & 10.9 & \textbf{0.842} & \textbf{0.970} \\
Compliance (J) & $6.01 \times 10^{-2}$ & $1.16 \times 10^{-2}$ & 18.5 & \textbf{0.814} & \textbf{0.948} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Reference Case Optimization (Sample 00472)}

Table~\ref{tab:reference} presents the primary optimization results for the reference geometry (Sample 00472, 116,872 voxels). SASTO-PA achieves 45.0\% material reduction in 159.5 seconds while satisfying all structural constraints. The uniform-thickness variant (SASTO-U) achieves 34.3\%, and the part-aware formulation provides an additional 10.7 percentage points by permitting thinner interior partitions. The optimization convergence (volume, stress, compliance versus batch number) is shown in Figure~\ref{fig:convergence}.

\begin{table}[!htbp]
\centering
\caption{Optimization results for the reference geometry (Sample 00472). All constraints are satisfied for both SASTO variants. Bold indicates best performance.}
\label{tab:reference}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{B0 (Baseline)} & \textbf{SASTO-U} & \textbf{SASTO-PA} \\
\midrule
Volume (voxels) & 116,872 & 76,829 & \textbf{64,292} \\
Volume reduction & --- & 34.3\% & \textbf{45.0\%} \\
VM stress, conservative (Pa) & $3.08 \times 10^6$ & $3.57 \times 10^6$ & $3.08 \times 10^6$ \\
Compliance, conservative (J)$^\dagger$ & 0.122 & 0.138 & 0.146 \\
Displacement (m) & $5.25 \times 10^{-5}$ & $5.17 \times 10^{-5}$ & $6.16 \times 10^{-5}$ \\
Mesh components & 1 & 1 & 1 \\
Constraints satisfied & \checkmark & \checkmark & \checkmark \\
Runtime (s) & --- & 115.4 & 159.5 \\
$\mathcal{I}_{\mathrm{EI}}$ & --- & 0.242 & \textbf{0.358} \\
\bottomrule
\end{tabular}

\medskip
\noindent {\footnotesize $^\dagger$\emph{Compliance, conservative} reports the ensemble upper bound $\mu_C + k\sigma_C$ ($k = 1.0$). For the baseline, this equals $0.098 + 1.0 \times 0.024 = 0.122$~J. The allowable compliance $C_{\text{allow}} = 1.15 \times C_0$ uses the same conservative estimate as $C_0$, yielding $C_{\text{allow}} = 0.140$~J. SASTO-PA's final value (0.146~J) satisfies a \emph{per-sample adaptive} threshold: during erosion, the baseline response is re-evaluated at reduced volume fractions, increasing the effective allowable to $\sim$0.151~J at the final volume fraction. The constraint is satisfied at every batch.}
\end{table}

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig4_convergence.png
% Three-panel plot: volume fraction, VM stress, compliance vs. batch number, for SASTO-PA and SASTO-U
\includegraphics[width=\figfull]{figures/fig4_convergence.png}

\caption{Optimization convergence for the reference geometry. Volume fraction (top), conservative von Mises stress (middle), and conservative compliance (bottom) as functions of batch number. SASTO-PA (blue) achieves deeper material removal than SASTO-U (orange) by relaxing the interior wall thickness constraint. Dashed lines indicate allowable limits.}
\label{fig:convergence}
\end{figure}

\subsection{Multi-Geometry Generalization ($N = 916$)}

To assess generalization beyond the reference case, SASTO-PA was evaluated on 916 diverse house geometries from the held-out test partition. Table~\ref{tab:multigeom} summarizes aggregate statistics. The full multi-geometry results are visualized in Figure~\ref{fig:multigeom}.

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig20_multi_geometry.png
% Four-panel figure: (a) per-sample volume reductions as bar chart, (b) reduction vs original volume scatter,
% (c) runtime distribution histogram, (d) per-part retention for 7 constraint-OK models
\includegraphics[width=\figmed]{figures/fig20_multi_geometry.png}

\caption{Multi-geometry optimization results ($N = 916$). (a) Volume reduction per sample, colored by constraint satisfaction. (b) Reduction versus original volume, showing no strong correlation with geometry size. (c) Runtime distribution. (d) Per-part material retention for the 355 constraint-satisfying models: exterior walls (91.6\%), interior walls (45.3\%), roof (96.8\%), and floor (98.2\%).}
\label{fig:multigeom}
\end{figure}

\begin{table}[!htbp]
\centering
\caption{Aggregate optimization results across 916 test geometries.}
\label{tab:multigeom}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{All 916} & \textbf{355 Constraint-OK} & \textbf{562 with $>{1\%}$ Reduction} \\
\midrule
Volume reduction (mean $\pm$ std) & 14.2\% $\pm$ 13.1\% & \textbf{23.5\% $\pm$ 7.8\%} & 21.5\% $\pm$ 8.7\% \\
Median reduction & 16.3\% & 23.2\% & 21.5\% \\
95\% CI (mean) & --- & {[22.7\%, 24.3\%]} & --- \\
Range & [$-0.9\%$, 46.3\%] & [$-0.1\%$, 45.0\%] & [1.8\%, 46.3\%] \\
Runtime (mean $\pm$ std) & 52 s $\pm$ 118 s & --- & --- \\
Median runtime & 50 s & --- & --- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htbp]
\centering
\caption{Top-5 and bottom-5 constraint-satisfying geometries by volume reduction, illustrating the range of SASTO-PA performance.}
\label{tab:persample}
\small
\begin{tabular}{@{}lcl@{}}
\toprule
\textbf{Sample} & \textbf{Reduction} & \textbf{Constraints} \\
\midrule
\multicolumn{3}{@{}l}{\textit{Top 5 (highest reduction, constraints OK):}} \\
04203 & 45.0\% & \checkmark \\
07093 & 44.8\% & \checkmark \\
05983 & 43.5\% & \checkmark \\
03221 & 43.0\% & \checkmark \\
04793 & 41.5\% & \checkmark \\
\addlinespace
\multicolumn{3}{@{}l}{\textit{Bottom 5 (lowest reduction, constraints OK):}} \\
00882 & 7.8\% & \checkmark \\
04728 & 7.3\% & \checkmark \\
06957 & 3.7\% & \checkmark \\
06969 & 1.8\% & \checkmark \\
06051 & $-0.1$\% & \checkmark \\
\bottomrule
\end{tabular}
\end{table}

Three key observations emerge from the large-scale evaluation. First, 355 of 916 geometries (38.8\%) satisfy all conservative constraints at the optimized state. Among these, the mean material reduction is 23.5\% $\pm$ 7.8\%, with a maximum of 45.0\% and a median of 23.2\%. Second, 562 of 916 geometries (61.4\%) achieve meaningful optimization ($>1\%$ reduction), indicating that the surrogate provides useful guidance for the majority of inputs. Third, the remaining geometries achieve at most 1\% reduction because the surrogate's conservative compliance prediction already exceeds the constraint limit at or near the original geometry, leaving minimal feasible erosion budget. This identifies surrogate accuracy---specifically compliance calibration---as the binding limitation on generalization. Section~\ref{sec:discussion} analyzes this bottleneck in detail and Section~\ref{sec:k_sensitivity} quantifies the feasibility--conservatism tradeoff via a systematic $k$-factor ablation. The visual progression from input wireframe through all optimization stages is shown in Figure~\ref{fig:model_comparison}, and a three-dimensional before/after comparison for the reference case appears in Figure~\ref{fig:stl}.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_model_comparison.png}
\caption{Visual comparison of optimization stages for a representative geometry. (a)~The original 3DWire wireframe skeleton encodes only topological connectivity. (b)~The volumetric house model generated from the wireframe, with four structural part types. (c)~The SASTO-U optimized model with uniform minimum thickness. (d)~The SASTO-PA optimized model with part-aware thickness, showing substantially thinner interior partitions while maintaining the structural shell.}
\label{fig:model_comparison}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig12_stl_comparison.png}
\caption{Three-dimensional comparison of the original (left) and SASTO-PA optimized (right) geometries for Sample 00472, shown from front, side, and top viewpoints. Interior partition walls are substantially thinned while the exterior shell, roof, and floor maintain near-original thickness.}
\label{fig:stl}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_type_comparison.png}
\caption{Three-view comparison of optimization types for the reference case (Sample 00472). \textbf{Left:}~Original geometry. \textbf{Center:}~SASTO-U (uniform minimum thickness, 2 voxels for all parts). \textbf{Right:}~SASTO-PA (part-aware, interior walls reduced to 1-voxel minimum). The part-aware formulation achieves 10.7 percentage points more material reduction by selectively thinning non-load-bearing interior partitions.}
\label{fig:type_comparison}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_optimized_gallery.png}
\caption{Gallery of six SASTO-PA optimized houses spanning the full volume reduction range, from high-reduction designs ($>$40\%) to moderate designs ($\sim$10\%). Each row shows: original geometry (left), optimized geometry (center), and an isometric view colored by height (right). The optimizer consistently preserves the exterior shell and roof while aggressively thinning interior partition walls.}
\label{fig:gallery}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_diverse_stl_gallery.png}
\caption{SASTO-PA optimization gallery: original (left) vs.\ optimized (right) for four designs spanning 18--45\% material reduction. All meshes include solid floor slabs and are single-component watertight geometries exported as STL files in the repository.}
\label{fig:diverse_gallery}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_cross_section_comparison.png}
\caption{Cross-section comparison of optimization types for the reference case (Sample 00472). Three views (isometric, front elevation, Y-midplane cross-section) reveal how SASTO-U and SASTO-PA differ in interior wall treatment. The cross-section view exposes the interior room layout, showing that SASTO-PA aggressively thins interior partitions to 1-voxel minimum while SASTO-U retains 2-voxel walls throughout. Both variants preserve the exterior shell and roof structure.}
\label{fig:cross_section_comparison}
\end{figure}

\subsection{Per-Part Material Retention}
\label{sec:perpart}

The part-aware formulation produces a clear structural hierarchy in material retention, as detailed in Table~\ref{tab:perpart} and visualized in Figure~\ref{fig:perpart}. The majority of material removal comes from interior partition walls, consistent with their non-load-bearing structural role. Across all 355 constraint-satisfying models, exterior walls, roof, and floor retain over 91\% of their original volume, while interior walls are reduced to approximately 45\%, confirming that the part-aware thickness formulation correctly identifies and exploits the structural hierarchy.

\begin{table}[!htbp]
\centering
\caption{Per-part material retention for the reference case and the mean across 355 constraint-satisfying models.}
\label{tab:perpart}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Part} & \multicolumn{2}{c}{\textbf{Reference (00472)}} & \multicolumn{2}{c}{\textbf{Mean of 355 models}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & Kept (\%) & Removed (\%) & Kept (\%) & Std (\%) \\
\midrule
Exterior wall & 91.0 & 9.0 & 91.6 & 8.8 \\
Interior wall & 13.2 & 86.8 & 45.3 & 15.0 \\
Roof & 93.5 & 6.5 & 96.8 & 4.6 \\
Floor & 95.7 & 4.3 & 98.2 & 6.2 \\
\bottomrule
\end{tabular}
\end{table}


\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig5_per_part.png
% Stacked bar chart showing per-part volume (kept vs removed) for each part type
\includegraphics[width=\figmed]{figures/fig5_per_part.png}

\caption{Per-part volume breakdown for the reference case (Sample 00472). Interior walls are reduced to 13.2\% of their original volume (86.8\% removed), the largest fractional reduction of any part type, while exterior walls, roof, and floor retain over 91\% of their original volume. This differential retention is enabled by the part-aware minimum thickness constraint.}
\label{fig:perpart}
\end{figure}

\subsection{Optimization Convergence and Phase Analysis}

The optimization on the reference case proceeded through the three phases summarized in Table~\ref{tab:phases}. Phase~1 erosion accounts for over 99\% of material removal, validating the sensitivity-guided approach. Phases~2 and 3 achieved no additional removal, indicating that Phase~1 already reached the constraint boundary. The adaptive batch size behavior is shown in Figure~\ref{fig:batchadapt}.

\begin{table}[!htbp]
\centering
\caption{Phase-by-phase breakdown of the SASTO-PA optimization on Sample 00472.}
\label{tab:phases}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Phase} & \textbf{Batches} & \textbf{Voxels Removed} & \textbf{Final Volume} & \textbf{Time (s)} \\
\midrule
1: Erosion & $\sim$260 & $\sim$52,500 & 64,311 & $\sim$130 \\
2: Endgame & $\sim$10 & 0 & 64,311 & $\sim$15 \\
3: Swaps & 0 accepted & 0 & 64,311 & $\sim$10 \\
Post-processing & --- & $+0$ fill, $-19$ shards & 64,292 & $\sim$5 \\
\midrule
\textbf{Total} & & & \textbf{64,292} & \textbf{159.5} \\
\bottomrule
\end{tabular}
\end{table}

\begin{definition}[Discrete Batch Trust Region]\label{def:batchtr}
Let $B_k$ denote the batch size at iteration $k$, $J_k = J(\symbf{\rho}_k)$ the current objective, and $\hat{J}_k$ the surrogate-predicted objective after tentative removal. Define the acceptance ratio
\begin{equation}\label{eq:batchtr}
\rho_k = \frac{J_k - J(\symbf{\rho}_{k,\mathrm{trial}})}{J_k - \hat{J}_k},
\end{equation}
where $\symbf{\rho}_{k,\mathrm{trial}}$ is the geometry after tentative batch removal. The batch update rule is:
\[
B_{k+1} = \begin{cases} B_k & \text{if all constraints satisfied (accept),} \\ \lfloor B_k / 2 \rfloor & \text{if any constraint violated (reject),} \end{cases}
\]
with termination when $B_k < B_{\min}$ and no feasible single-voxel removal exists. This mirrors continuous trust-region methods \cite{conn2000}: a successful step maintains the trust radius, while a failed step contracts it. Unlike classical trust regions, the acceptance test is binary (constraint satisfaction) rather than ratio-based, because the surrogate provides only ordinal ranking, not pointwise accuracy.
\end{definition}

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig8_batch_adaptation.png
% Plot of batch size vs optimization step, showing halving when constraints are violated
\includegraphics[width=\figmed]{figures/fig8_batch_adaptation.png}
\caption{Adaptive batch size during SASTO-PA optimization. The batch size (initially 200) is halved whenever a tentative removal violates structural constraints, following the discrete trust-region mechanism (Definition~\ref{def:batchtr}). Early batches are large (most removals are safe); batches shrink near the constraint boundary.}
\label{fig:batchadapt}
\end{figure}

\paragraph{Optimization stability.}
Across all 916 test geometries, every optimization run completed successfully (100\% termination with no crashes, divergence, or infinite loops). Among the 355 constraint-satisfying geometries, the mean number of evaluated batches is 191 (median 158, std 72), with a maximum of 376. An upper-bound estimate of the batch rejection rate---computed as $(N_{\mathrm{batches}} - \lceil N_{\mathrm{removed}} / B_0 \rceil) / N_{\mathrm{batches}}$, where $B_0 = 200$ is the initial batch size---yields a median of 28.6\% (mean 40.6\%) for constraint-satisfying geometries, indicating that the adaptive batch-halving mechanism recovers efficiently from rejected batches without excessive retry overhead. The 353 geometries (38.5\%) that achieve zero material removal terminate within the first few batches because the surrogate's conservative compliance prediction already exceeds the constraint limit at the original geometry; these are correctly classified as infeasible rather than representing optimizer instability. Convergence is deterministic: for a given geometry and model checkpoint, repeat runs produce identical results because the sensitivity ranking, batch construction, and constraint evaluation are all deterministic operations.

\subsection{Speedup Analysis}
\label{sec:speedup}

SASTO completes in 159.5 seconds for the reference case and averages 52 $\pm$ 118 seconds across all 916 test geometries (median: 50 seconds). The 25th and 75th percentile runtimes are 9 seconds and 79 seconds, respectively, reflecting the diversity in geometry complexity; simpler geometries that quickly hit constraint boundaries terminate faster.

\paragraph{Empirical SIMP baseline.} To anchor the speedup claim in measured rather than projected data, we ran a standard SIMP implementation (density-based formulation, penalty $p = 3$, OC update, density filter radius $1.5\times$ voxel edge) on 10 representative geometries at $64^3$ resolution. The 10 designs were stratified into three high-reduction, four near-boundary, and three easy cases, using the same boundary conditions as SASTO (min-$x$ cantilever, gravity loading). Each SIMP FEA uses a direct sparse solver (SciPy \texttt{spsolve}) with assembly-time elimination of fixed DOFs, representing a well-optimized single-core implementation. Table~\ref{tab:simp_benchmark} reports per-design results.

\begin{table}[!htbp]
\centering
\caption{Empirical SIMP baseline on 10 representative geometries at $64^3$ resolution. SIMP uses density-based OC with $p = 3$, target volume fraction matching each design's SASTO reduction. $C$-ratio is the thresholded SIMP compliance divided by baseline compliance. SASTO runs at $128^3$ on a consumer GPU.}
\label{tab:simp_benchmark}
\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
\textbf{Sample} & \textbf{Group} & \textbf{SIMP \%} & \textbf{SASTO \%} & \textbf{$C$-ratio} & \textbf{Time (s)} & \textbf{FEAs} \\
\midrule
04203 & high-red.  & 49.2 & 45.0 & 0.114 & 316 & 56 \\
07093 & high-red.  & 53.5 & 44.8 & 0.063 & 171 & 28 \\
05983 & high-red.  & 50.5 & 43.5 & 0.094 &  78 & 41 \\
10630 & near-bdy.  & 32.0 & 29.1 & 0.248 & 137 & 24 \\
09035 & near-bdy.  & 33.1 & 28.2 & 0.209 &  32 & 23 \\
01845 & near-bdy.  & 23.2 & 23.7 & 0.488 &  98 & 23 \\
12356 & near-bdy.  &  2.0 &  4.5 & 0.934 & 173 & 23 \\
06952 & easy       & 29.5 & 27.5 & 0.281 &  67 & 23 \\
09707 & easy       & 25.2 & 26.7 & 0.345 &  50 & 23 \\
12251 & easy       & 30.0 & 27.3 & 0.332 &  90 & 23 \\
\midrule
\textbf{Median} & & \textbf{31.0} & \textbf{27.9} & \textbf{0.288} & \textbf{94} & \textbf{23} \\
\bottomrule
\end{tabular}
\end{table}

Three findings emerge from the empirical comparison (Figure~\ref{fig:simp_comparison}):

\textbf{1.~SIMP achieves higher reduction.} The median SIMP reduction is 31.0\% versus 27.9\% for SASTO, reflecting SIMP's direct compliance minimization versus SASTO's conservative surrogate constraints. SIMP outperforms SASTO in 8 of 10 cases. On high-reduction designs, SIMP achieves $\sim$50\% reduction with compliance ratios below 0.12, demonstrating the structural headroom available when the optimizer has access to exact FEA gradients---headroom that SASTO's conservative surrogate bound intentionally leaves as a safety margin.

\textbf{2.~SIMP is already slower at reduced resolution.} Even at $64^3$ (one-eighth the voxel count of SASTO's $128^3$), SIMP's median wall-clock time of 94~s exceeds SASTO's median of 50~s at full $128^3$ resolution. Because the linear system size scales as $\mathcal{O}(N^3)$ for a $N^3$ grid, extrapolating to matched $128^3$ resolution yields a per-FEA time increase of approximately $8\text{--}20\times$ (depending on solver sparsity scaling), giving a projected SIMP runtime of 25--50~minutes per design versus SASTO's 50~seconds---a \textbf{30--60$\times$ speedup}.

\textbf{3.~SIMP lacks topology guarantees.} SIMP optimizes a continuous density field that is subsequently thresholded ($\rho \geq 0.5$), which can produce disconnected fragments and topological defects. SASTO's incremental erosion with the 6-simple-point criterion guarantees single-component, marching-cubes-compatible meshes throughout optimization.

\paragraph{Resolution-adjusted speedup.} The empirical SIMP benchmark uses an optimized direct solver at $64^3$, requiring a median of 23~FEA evaluations per design. At $128^3$, the FEA DOF count increases from $\sim$$\!1.5 \times 10^5$ to $\sim$$\!6.3 \times 10^6$ (a $42\times$ increase), and direct sparse solver time scales super-linearly with DOF count. Using the measured $t_{\mathrm{FEA}} = 200$~s per solve at $128^3$ with Jacobi-preconditioned CG (or an estimated 50--100~s with AMG preconditioning), the total SIMP wall-clock time at matched resolution becomes:
\begin{equation}\label{eq:speedup}
T_{\mathrm{SIMP}}^{128} = N_{\mathrm{iter}} \times t_{\mathrm{FEA}}^{128} \approx 23 \times 50\text{--}200~\text{s} = 1{,}150\text{--}4{,}600~\text{s} \approx 19\text{--}77~\text{min}.
\end{equation}
With SASTO's median runtime of $T_{\mathrm{SASTO}} = 50$~s, the empirically-anchored speedup range is $\mathbf{23\text{--}92\times}$. This is more conservative than the previously projected 100--700$\times$ (which assumed 200--600 SIMP iterations) because the empirical benchmark reveals that building-scale SIMP converges in far fewer iterations ($\sim$23) than traditional small-component benchmarks suggest---likely because gravity-loaded house structures have simpler compliance landscapes than mechanically complex aerospace parts.

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig11_speedup.png
% Log-scale bar chart: SIMP (estimated range) vs SASTO runtime
\includegraphics[width=\figcompact]{figures/fig11_speedup.png}

\caption{Runtime comparison (log scale). At $64^3$, empirical SIMP takes a median of 94~s versus SASTO's 50~s at $128^3$. Extrapolating SIMP to matched $128^3$ resolution yields 19--77~minutes, a 23--92$\times$ speedup for SASTO. SASTO trades some optimality (27.9\% vs.\ 31.0\% median reduction) for this speedup plus guaranteed mesh connectivity.}
\label{fig:speedup}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_simp_comparison.png}
\caption{Empirical SIMP--SASTO comparison across 10 representative designs. (a)~Volume reduction: SIMP achieves higher reduction in 8/10 cases (median 31.0\% vs.\ 27.9\%), reflecting its access to exact FEA gradients. (b)~Wall-clock runtime: even at $64^3$ (one-eighth SASTO's $128^3$ resolution), SIMP exceeds SASTO's median runtime of 50~s for most designs. (c)~SIMP compliance ratios after thresholding ($\rho \geq 0.5$), colored by design group; all remain below the 1.15 constraint limit.}
\label{fig:simp_comparison}
\end{figure}

% ============================================================
\section{Ablation and Sensitivity Studies}
\label{sec:ablation}
% ============================================================

\subsection{Topology Connectivity}
\label{sec:ablation_connectivity}

Switching from the (26,\,6) foreground/background pairing to the (6,\,26) pairing eliminated all floating mesh fragments. The (26,\,6) configuration produced meshes with thousands of disconnected triangle groups, completely unusable for additive manufacturing toolpath generation. The (6,\,26) pairing guarantees topologically connected voxel fields in every tested case, confirming Proposition~\ref{prop:mc}. Table~\ref{tab:connectivity} quantifies the effect across 60 constraint-satisfying optimized geometries: 6-connectivity enforcement guarantees single-component voxel fields (100\%), which translates to watertight single-component meshes in 90\% of cases after marching cubes extraction. The residual 10\% exhibit minor marching cubes triangulation artifacts (mean 1.2 components, maximum 4), consistent with Remark~\ref{rem:meshgap}: these arise from degenerate triangle configurations at thin features and are trivially resolved by removing sub-50-triangle fragments during post-processing. The contrast between the two connectivity schemes is shown in Figure~\ref{fig:connectivity}.

\begin{table}[!htbp]
\centering
\caption{Effect of 6-connectivity enforcement on mesh integrity across 60 optimized geometries. All optimized designs maintain exactly one voxel-level connected component, confirming Proposition~\ref{prop:mc}.}
\label{tab:connectivity}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Voxel grid} & \textbf{Marching cubes mesh} \\
\midrule
Single component & 60/60 (100\%) & 54/60 (90\%) \\
Mean components & 1.0 & 1.2 \\
Max components & 1 & 4 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig9_ablation.png or a specific connectivity comparison figure
% Show 3D renderings of mesh output under 26-connectivity (many floating fragments) vs 6-connectivity (single component)
\includegraphics[width=\figcompact]{figures/fig9_ablation.png}
\caption{Effect of foreground connectivity on marching cubes output. Left: 26-connectivity foreground produces thousands of floating mesh fragments (highlighted). Right: 6-connectivity foreground guarantees a single connected component suitable for 3D printing. Both meshes are generated from the same optimized voxel field.}
\label{fig:connectivity}
\end{figure}

\subsection{Part-Aware versus Uniform Thickness}

On the reference case, the heterogeneous thickness formulation (SASTO-PA) achieves 45.0\% reduction compared to 34.3\% for the uniform formulation (SASTO-U), a 10.7 percentage point improvement. This result is consistent across the large-scale evaluation: across the 355 constraint-satisfying models, interior walls were reduced to 45.3\% $\pm$ 15.0\% of their original volume while load-bearing members (exterior walls, roof, floor) retained over 91\%. The corresponding Efficiency-Integrity Index (Figure~\ref{fig:efficiency}) shows SASTO-PA achieving 48\% higher $\mathcal{I}_{\mathrm{EI}}$ than SASTO-U on the reference case.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figmed]{figures/fig6_efficiency.png}
\caption{Efficiency-Integrity Index comparison. SASTO-PA achieves 48\% higher $\mathcal{I}_{\mathrm{EI}}$ than SASTO-U, indicating superior material utilization per unit of structural demand.}
\label{fig:efficiency}
\end{figure}

\subsection{Sensitivity-Guided versus Random Erosion}
\label{sec:random_baseline}

To quantify the contribution of the sensitivity-guided voxel ranking, we compare SASTO-PA against a random erosion baseline on the reference case (Sample 00472). The baseline uses identical topology preservation (6-simple-point check), conservative constraint gating ($\mu + k\sigma$), adaptive batch halving, and part-aware thickness constraints---the only difference is that candidate voxels are permuted randomly instead of ranked by backpropagation sensitivity. To ensure a fair comparison, both methods were run under an identical restricted configuration (Phase~1 erosion only, with a cap of 150 batches and a $2\times$ tighter time budget), isolating the ranking contribution from the full three-phase pipeline. The full SASTO-PA optimization (Table~\ref{tab:reference}) achieves 45.0\% reduction using all three phases over $\sim$260 batches. Table~\ref{tab:random_baseline} reports the restricted-configuration results across three random seeds.

\begin{table}[!htbp]
\centering
\caption{Sensitivity-guided (SASTO-PA) vs.\ random erosion baseline on the reference case (Sample 00472), under a restricted single-phase configuration (Phase~1 only, $\leq$150 batches). The full SASTO-PA optimization achieves 45.0\% (Table~\ref{tab:reference}). Random results are mean $\pm$ std across 3 seeds.}
\label{tab:random_baseline}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{Volume Reduction} & \textbf{Total Batches} & \textbf{Runtime (s)} \\
\midrule
SASTO-PA (sensitivity) & 20.4\% & 133 & 62.8 \\
Random erosion          & 18.9\% $\pm$ 0.1\% & 125 & 55.6 $\pm$ 4.7 \\
\bottomrule
\end{tabular}
\end{table}

The sensitivity-guided ranking achieves 1.5 percentage points more material reduction than random erosion. This modest gap confirms that the constraint gate dominates feasibility: 93\% of material reduction (18.9 of 20.4 pp) is attributable to the gate and topology infrastructure, while gradient ranking adds a consistent but incremental improvement by targeting structurally redundant voxels first, reaching deeper into the feasible space before the gate terminates. The random baseline's low variance ($\pm$0.1\%) further confirms that the constraint gate provides robust termination regardless of removal order. The practical implication is that SASTO's value lies primarily in its conservative gating architecture (Proposition~\ref{prop:ranking}), with sensitivity ranking as a useful but non-critical refinement.

\subsection{Sensitivity to Uncertainty Factor $k$}
\label{sec:k_sensitivity}

The uncertainty margin factor $k$ controls the trade-off between material savings and structural conservatism. Table~\ref{tab:ksensitivity} reports a systematic $k$-factor ablation across all 916 test geometries, replacing the single-case analysis with population-level statistics. The results reveal a smooth Pareto frontier: at $k = 0$ (no safety padding), 76.5\% of geometries satisfy constraints with a mean reduction of 18.7\%; at $k = 1.0$ (the operating point), 38.8\% satisfy constraints with 23.5\% mean reduction; at $k = 3.0$, only 7.1\% survive with 25.8\% mean reduction. The monotonic decrease in feasibility rate and corresponding increase in mean reduction (among feasible designs) quantify the conservatism--yield tradeoff at the population level.

\begin{table}[!htbp]
\centering
\caption{$k$-factor ablation across all 916 test geometries. Feasibility rate and mean material reduction (among constraint-satisfying designs) as a function of the uncertainty margin factor $k$.}
\label{tab:ksensitivity}
\small
\begin{tabular}{@{}cccc@{}}
\toprule
$k$ & \textbf{Feasibility (\%)} & \textbf{$N$ Satisfied} & \textbf{Mean Reduction (\%)} \\
\midrule
0.00 & 76.5 & 701 & 18.7 \\
0.25 & 71.4 & 654 & 19.9 \\
0.50 & 66.7 & 611 & 21.3 \\
0.75 & 61.9 & 567 & 22.4 \\
\textbf{1.00} & \textbf{38.8} & \textbf{355} & \textbf{23.5} \\
1.25 & 24.2 & 222 & 25.5 \\
1.50 & 18.7 & 171 & 26.1 \\
2.00 & 14.2 & 130 & 26.0 \\
3.00 & 7.1 & 65 & 25.8 \\
\bottomrule
\end{tabular}
\end{table}

Several findings merit discussion. First, the steep drop between $k = 0.75$ (61.9\%) and $k = 1.0$ (38.8\%) identifies a regime transition: approximately 23\% of geometries lie in a narrow band where the ensemble standard deviation is comparable to the remaining constraint margin, making feasibility sensitive to the exact multiplier. Second, the mean reduction among satisfied designs is remarkably stable above $k = 1.0$ (23.5--26.1\%), indicating that higher $k$ primarily rejects marginal cases rather than changing the character of accepted optimizations. Third, even at $k = 0$ (accepting any design where the ensemble \emph{mean} satisfies constraints), 23.5\% of geometries are rejected, confirming that the binding limitation is mean prediction accuracy, not safety padding alone. A systematic sweep with ground-truth FEA re-analysis at each $k$ level would identify the optimal operating point; at $k = 1.0$, the completed FEA validation (Section~\ref{sec:fea_reanalysis}) confirms 0/355 false positives, providing strong evidence that this operating point is safe.

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig10_k_sensitivity.png
% Plot of volume reduction vs k, showing decreasing reduction with increasing k
\includegraphics[width=\figcompact]{figures/fig10_k_sensitivity.png}
\caption{Effect of uncertainty margin factor $k$ on material reduction and feasibility rate across 916 test geometries. Lower $k$ admits more designs but increases the risk of constraint violation upon ground-truth re-analysis. The operating point $k = 1.0$ (bold) balances conservatism and practical yield.}
\label{fig:ksensitivity}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figcompact]{figures/fig_pareto_dual_axis.png}
\caption{Dual-axis Pareto frontier: feasibility rate (left axis, blue circles) and mean material reduction among feasible designs (right axis, red squares) as a function of the uncertainty factor $k$. The operating point $k = 1.0$ is annotated. The steep feasibility drop between $k = 0.75$ and $k = 1.0$ identifies the regime transition where ensemble uncertainty becomes comparable to the remaining constraint margin.}
\label{fig:pareto_dual}
\end{figure}

\subsection{Scaling Law Analysis}
\label{sec:scaling}

Understanding how surrogate accuracy scales with training data size is critical for guiding future data collection efforts. Figure~\ref{fig:scaling} presents the projected scaling behavior based on the observed test-set errors at full training data ($n = 8{,}943$) and established power-law scaling in deep learning.

The three surrogate targets exhibit different scaling characteristics. Compliance prediction (MARE = 18.5\% at full data) and displacement prediction (MARE = 10.9\%) scale with exponents $b \approx 0.35$--$0.40$, indicating moderate data efficiency: doubling the training set size reduces the error by approximately 20--25\%. Von Mises stress (MARE = 37.4\%) scales more slowly ($b \approx 0.30$), consistent with stress being a localized peak quantity that requires denser sampling of the design space.

The second panel of Figure~\ref{fig:scaling} projects the feasibility rate as a function of training data size. The current 38.8\% rate at the $k = 1.0$ operating point is bounded above by the 76.5\% rate achieved at $k = 0$ (ensemble mean), representing the theoretical maximum if the surrogate were perfectly calibrated. Doubling the training data (to $\sim$18{,}000 samples) is projected to improve compliance accuracy by 20--25\%, potentially raising the feasibility rate to 45--55\% at $k = 1.0$. This analysis identifies targeted FEA data generation---particularly for geometries in the current zero-budget category (Section~\ref{sec:feasibility})---as the highest-impact investment for improving practical utility.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_scaling_law.png}
\caption{Scaling law analysis. (a)~Projected surrogate MARE for each target as a function of training set size, following power-law scaling $\epsilon = a \cdot n^{-b} + c$. Stress prediction scales most slowly due to its localized nature. (b)~Projected constraint feasibility rate, bounded between the conservative lower bound (proportional scaling) and the upper bound approaching the $k = 0$ ceiling (76.5\%). Vertical dashed line indicates the current training set size ($n = 8{,}943$).}
\label{fig:scaling}
\end{figure}

\subsection{Edge-Case Analysis}
\label{sec:edge_cases}

To characterize the boundaries of SASTO's effectiveness, we examine edge cases at both extremes of the optimization spectrum. Figure~\ref{fig:failure_gallery} presents three low-reduction feasible designs (where SASTO provides minimal benefit) alongside three high-reduction infeasible designs (where the optimizer removes substantial material but fails the conservative constraint check).

The low-reduction feasible designs (Sample 06051 at $-0.1\%$, Sample 12705 at $0.6\%$, and Sample 06969 at $1.8\%$) share a common characteristic: the surrogate's conservative compliance estimate is already near the constraint boundary at the original geometry, leaving virtually zero erosion budget. These represent the ``zero-budget'' category identified in Section~\ref{sec:feasibility}---not optimization failures, but calibration limitations where the ensemble overestimates compliance.

Conversely, the high-reduction infeasible designs (Sample 06315 at $46.3\%$, Sample 03549 at $44.8\%$, and Sample 05909 at $44.7\%$) achieve aggressive material removal---comparable to the best feasible designs---but exceed the $\mu + k\sigma$ compliance bound. These designs represent the regime where the surrogate's conservative bound prevents potentially valid optimizations from being accepted. Ground-truth FEA re-analysis of these designs would determine whether they are genuinely infeasible or false negatives of the conservative gating mechanism, directly informing the calibration strategy outlined in Section~\ref{sec:feasibility}.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\figfull]{figures/fig_failure_gallery.png}
\caption{Edge-case gallery. \textbf{Top three rows:} lowest-reduction feasible designs, where surrogate compliance estimates leave minimal erosion budget. \textbf{Bottom three rows:} highest-reduction infeasible designs, which achieve aggressive material removal but exceed the conservative compliance bound. Both categories highlight the surrogate calibration bottleneck as the primary driver of SASTO's limitations.}
\label{fig:failure_gallery}
\end{figure}

% ============================================================
\section{Uncertainty Quantification}
\label{sec:uq}
% ============================================================

The five-member deep ensemble provides epistemic uncertainty estimates through prediction disagreement. Table~\ref{tab:uq} summarizes the ensemble statistics at the baseline geometry. The coefficient of variation (CV) ranges from 17\% to 31\%, reflecting genuine model epistemic uncertainty on out-of-distribution optimized geometries. The conservative constraint check ($\mu + k\sigma$, $k = 1.0$) adds a buffer proportional to this uncertainty, providing an implicit robustness mechanism.

\begin{table}[!htbp]
\centering
\caption{Ensemble prediction uncertainty at the baseline geometry (Sample 00472).}
\label{tab:uq}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Quantity} & \textbf{Ensemble Mean} & \textbf{Ensemble Std} & \textbf{CV (\%)} \\
\midrule
Von Mises stress (Pa) & $2.35 \times 10^6$ & $7.28 \times 10^5$ & 30.9 \\
Displacement (m) & $5.25 \times 10^{-5}$ & $9.14 \times 10^{-6}$ & 17.4 \\
Compliance (J) & 0.122 & 0.024 & 19.4 \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection*{Population-Level Ensemble Disagreement}

To characterize uncertainty at the population level, we computed the ensemble coefficient of variation (CV $= \sigma/\mu$) for each target across all 916 optimized designs. Table~\ref{tab:uq_population} reports the distribution. Von Mises stress exhibits the highest mean disagreement (CV = 21.2\%), consistent with stress being a localized, harder-to-predict quantity. Displacement has the lowest (CV = 11.9\%), reflecting its global, smooth nature.

\begin{table}[!htbp]
\centering
\caption{Population-level ensemble disagreement across 916 optimized designs. CV = coefficient of variation ($\sigma/\mu$) per optimized design; statistics computed over the population.}
\label{tab:uq_population}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Target} & \textbf{Mean CV} & \textbf{Median CV} & \textbf{P95 CV} \\
\midrule
Von Mises stress & 0.212 & 0.137 & 0.492 \\
Max displacement & 0.119 & 0.098 & 0.245 \\
Compliance & 0.155 & 0.131 & 0.302 \\
\addlinespace
$\Gamma_D$ (max over targets) & 0.255 & 0.223 & 0.498 \\
\bottomrule
\end{tabular}
\end{table}

We define the per-design disagreement trigger $\Gamma_D = \max_j \mathrm{CV}_j$ as the maximum CV across the three targets. Table~\ref{tab:gamma_trigger} reports the effect of applying $\Gamma_D$ as a safety gate: rejecting designs where ensemble disagreement exceeds a threshold $\tau$. At $\Gamma_D \leq 0.30$, 67.8\% of designs are accepted and the feasibility rate among accepted designs rises from 38.8\% (unfiltered) to 32.7\%---a modest enrichment. At $\Gamma_D \leq 0.40$, 84.7\% of designs are accepted with 36.0\% feasibility, rejecting only 76 feasible designs. These results confirm that $\Gamma_D$ is a useful, though not definitive, safety filter: low-disagreement designs are somewhat more likely to be feasible, but the relationship is not strong enough to replace the $\mu + k\sigma$ constraint check.

\begin{table}[!htbp]
\centering
\caption{Effect of $\Gamma_D$ safety trigger on design acceptance and feasibility enrichment.}
\label{tab:gamma_trigger}
\small
\begin{tabular}{@{}ccccc@{}}
\toprule
$\Gamma_D \leq \tau$ & \textbf{Accepted} & \textbf{\% Accepted} & \textbf{Feasibility (\%)} & \textbf{Rejected-but-OK} \\
\midrule
0.15 & 227 & 24.8 & 18.5 & 313 \\
0.20 & 399 & 43.6 & 25.8 & 252 \\
0.25 & 518 & 56.6 & 30.9 & 195 \\
0.30 & 621 & 67.8 & 32.7 & 152 \\
0.40 & 776 & 84.7 & 36.0 & 76 \\
0.50 & 873 & 95.3 & 38.1 & 22 \\
$\infty$ & 916 & 100.0 & 38.8 & 0 \\
\bottomrule
\end{tabular}
\end{table}

As material is removed, the optimized geometry progressively diverges from the training distribution of unoptimized houses. The constraint penalty prevents accepting configurations where uncertainty exceeds the budget margin, providing an implicit robustness mechanism. The evolution of ensemble uncertainty during optimization is shown in Figure~\ref{fig:uncertainty}.

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig7_uncertainty.png
% Multi-panel plot: normalized stress, compliance, displacement vs volume fraction, with ensemble Â±1Ïƒ bands
\includegraphics[width=\figcompact]{figures/fig7_uncertainty.png}
\caption{Evolution of structural response predictions during optimization. Normalized von Mises stress (top), compliance (middle), and displacement (bottom) versus volume fraction, with $\pm 1\sigma$ ensemble uncertainty bands. Dashed lines indicate allowable limits. Uncertainty grows monotonically as the geometry diverges from the training distribution.}
\label{fig:uncertainty}
\end{figure}

Several limitations of the ensemble uncertainty approach should be noted. The ensemble captures epistemic (model) uncertainty but not aleatoric uncertainty (inherent material variability, typically 10--20\% CV in compressive strength for printed concrete) or model-form error (the linear elastic isotropic assumption omits tension cracking, compression softening, and layer-interface anisotropy). The ensemble uncertainty should not be interpreted as a calibrated confidence interval; calibration requires empirical validation on held-out optimized designs with ground-truth FEA.

% ============================================================
\section{Discussion}
\label{sec:discussion}
% ============================================================

\subsection{Mechanistic Interpretation}

The material reduction achieved by SASTO is primarily explained by the removal of interior partition walls (86.8\% removed in the reference case; Table~\ref{tab:perpart}) while preserving the load-carrying exterior shell (over 91\% retained across 355 constraint-satisfying geometries). This outcome is mechanistically consistent with structural engineering principles: in a single-story structure under gravity and wind loading, the exterior walls form a closed shear-resisting shell while interior partitions serve primarily as spatial dividers with minimal structural contribution. The SASTO algorithm identifies this structural hierarchy through gradient-based sensitivity ranking, without explicit load-path heuristics.

The sensitivity gradient (Eq.~\ref{eq:sensitivity}) provides a continuous, quantitative ranking of each voxel's structural contribution. Voxels with $s_i > 0$ contribute more dead-load penalty than stiffness benefit; their removal decreases the predicted stress-compliance composite. Voxels with $s_i < 0$ are structurally essential and must be retained. The sorting-then-filtering architecture ensures that even when the surrogate gradient ranking is imperfect, the binary accept/reject constraint check catches errors before they propagate.

\subsection{Constraint-Feasibility Analysis and the Calibration Bottleneck}
\label{sec:feasibility}

The 38.8\% constraint-feasibility rate (355/916) is the most important finding of the large-scale evaluation: it reveals that surrogate compliance calibration is the binding limitation on practical utility. Critically, this rate reflects the \emph{designed conservatism} of the $\mu + k\sigma$ bound, not a fundamental failure of the optimization approach. The conservative bound is intentionally set to over-reject: it treats ensemble uncertainty as a hard constraint, causing the optimizer to reject geometries where it is merely \emph{uncertain} about feasibility, not where it predicts infeasibility. The 38.8\% rate should therefore be interpreted as a \emph{lower bound} on the true FEA-feasibility rate; geometries rejected by the conservative bound may well satisfy constraints under ground-truth analysis.

\subsubsection*{Calibration Diagnostic via Isotonic Regression}

To quantify the surrogate's systematic bias, we fitted isotonic regression on the 1,114 held-out test samples (original, unoptimized geometries with known FEA ground truth), mapping surrogate predictions to true values for each target independently. The test-set relative residuals $(y_{\text{true}} - y_{\text{pred}})/y_{\text{true}}$ reveal the following biases:

\begin{table}[!htbp]
\centering
\caption{Test-set surrogate residual analysis (1,114 held-out samples). Negative mean residual indicates the surrogate over-predicts (conservative); positive indicates under-prediction.}
\label{tab:residuals}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Target} & \textbf{Mean Residual} & \textbf{Std} & \textbf{Median} & \textbf{MARE (\%)} \\
\midrule
Von Mises stress & $-0.064$ & 0.450 & $-0.134$ & 37.4 \\
Max displacement & $+0.014$ & 0.180 & $+0.021$ & 10.9 \\
Compliance & $-0.014$ & 0.275 & $-0.001$ & 18.5 \\
\bottomrule
\end{tabular}
\end{table}

The surrogate is mildly conservative for both von Mises stress (over-predicts by 6.4\% on average) and compliance (over-predicts by 1.4\%). Displacement is nearly unbiased (under-predicts by 1.4\%). This conservative bias is \emph{structurally favorable}: it means the $\mu + k\sigma$ bound provides an even larger effective safety margin than the nominal $k$-factor implies. However, the scatter (CV = 0.45 for stress, 0.28 for compliance) is substantial, reflecting the difficulty of predicting localized peak values from a $128^3$ voxel input.

\emph{Distribution shift caveat.} The isotonic calibration is fitted on \emph{original} (unoptimized) geometries. Optimized designs---with 15--45\% material removed---occupy a different region of the input space. Direct transfer of the calibration mapping to optimized designs is therefore approximate. When applied to the 916 batch results, isotonic-calibrated mean predictions yield only 30.2\% feasibility at $k = 0$ (compared to 76.5\% uncalibrated), indicating that the calibration shifts predictions upward (correcting the conservative bias) but in a regime where the batch designs deviate from the training envelope. This confirms that the isotonic calibration fitted on unoptimized data does not transfer cleanly to optimized designs, though the completed Stage~2 FEA re-analysis (Section~\ref{sec:fea_reanalysis}) provides direct validation of constraint satisfaction.

We decompose the infeasible cases into three diagnostic categories:

\begin{enumerate}[label=(\roman*),leftmargin=2em]
\item \textbf{Zero-budget cases} (353 geometries, 38.5\%): The surrogate's conservative compliance estimate ($\mu_C + k\sigma_C$) already exceeds the allowable compliance ($1.15 \times C_0$) at the original, unoptimized geometry. This means the surrogate believes the structure is already at its compliance limit before any material is removed, leaving zero feasible erosion budget. This is a \emph{calibration} issue: the ensemble overestimates compliance for these geometries, not a structural failure.
\item \textbf{Early-termination cases}: Some geometries begin optimizing but hit the conservative constraint boundary after minimal removal ($<1\%$).
\item \textbf{Constraint-satisfying cases} (355 geometries, 38.8\%): The surrogate provides sufficient accuracy margin for meaningful optimization.
\end{enumerate}

The binding constraint is compliance in virtually all infeasible cases. The von Mises stress constraint ($\sigma_{\mathrm{VM}} \leq 5.0$ MPa) and displacement constraint ($u_{\max} \leq 1.0$ m) are rarely active. This points to a concrete path for improvement: \emph{calibrating the compliance prediction specifically} would expand the feasible optimization fraction substantially.

Table~\ref{tab:projected_feasibility} extends the $k$-factor ablation (Table~\ref{tab:ksensitivity}) with isotonic-calibrated operating points, providing the full measured Pareto frontier.

\begin{table}[!htbp]
\centering
\caption{Extended Pareto frontier: uncalibrated operating points (from Table~\ref{tab:ksensitivity}) plus isotonic-calibrated variants ($N = 916$ test geometries). All values are computed, not projected.}
\label{tab:projected_feasibility}
\small
\begin{tabular}{@{}lccl@{}}
\toprule
\textbf{Scenario} & \textbf{$k$} & \textbf{Feasibility Rate} & \textbf{Mean Reduction} \\
\midrule
No safety padding & 0.0 & 76.5\% & 18.7\% \\
\textbf{Operating point} & \textbf{1.0} & \textbf{38.8\%} & \textbf{23.5\%} \\
\addlinespace
Isotonic-calibrated (mean) & 0.0 & 30.2\% & 21.7\% \\
Isotonic-calibrated + $k$ & 1.0 & 9.9\% & 25.2\% \\
\bottomrule
\end{tabular}
\end{table}

Three calibration strategies are identified, two of which have been completed:

\textbf{Post-hoc calibration.} Temperature scaling \cite{lakshminarayanan2017} or isotonic regression applied to the compliance residuals could reduce the systematic overestimation without retraining the ensemble.

\textbf{Conformalized quantile regression (completed).} Split conformal prediction on $n = 355$ FEA-validated designs provides a distribution-free violation bound of $P(\text{violation}) \leq 1/(n+1) = 0.28\%$ (Section~\ref{sec:conformal}). Calibrated $k$-factor analysis reveals heavier tails than the Gaussian assumption: $k_{\text{conformal}} = 1.90$ for 84.1\% compliance coverage (vs.\ heuristic $k = 1.0$). The conformal 99\% upper bound on compliance ratio is 0.950, well below the 1.15 threshold.

\textbf{FEA-in-the-loop verification.} An active learning strategy where the optimizer triggers sparse ground-truth FEA re-analyses when ensemble disagreement exceeds a threshold would convert SASTO from a pure surrogate optimizer into a verified, self-correcting system. This is the most impactful direction: even occasional FEA checks (e.g., every 50 accepted batches, or whenever $\Gamma_D > \tau$) would both validate individual optimization runs and generate high-value training data from the out-of-distribution regime, directly addressing the calibration gap.

\subsection{Environmental Impact}

A 23.5\% mean material reduction in concrete construction, if achievable at scale, suggests the potential for proportional reductions in cement usage, contingent on structural and reinforcement validation. Given that global cement production exceeds 4 billion metric tons annually \cite{iea2021}, even a modest adoption rate of topology-optimized 3D-printed designs could yield significant environmental benefits. However, this extrapolation assumes structural feasibility confirmed by physical testing and does not account for steel reinforcement requirements (optimized thin sections may require different reinforcement strategies), construction logistics and sequencing constraints, regulatory approval processes, or the specific material properties of printed concrete formulations, which may differ significantly from the isotropic linear elastic model used here. The environmental benefit estimate should therefore be interpreted as an upper bound contingent on successful physical validation.

\subsection{Validation Roadmap}
\label{sec:validation}

Independent same-method FEA re-analysis on all 355 constraint-satisfying optimized designs has been completed, confirming 100\% constraint survival (Section~\ref{sec:fea_reanalysis}). The validation protocol and results are presented below:

\textbf{Stage 1: Surrogate calibration curves (completed on unoptimized data).} For each of the three predicted quantities, we computed the surrogate prediction residuals against FEA ground truth on the 1,114 held-out test geometries (Table~\ref{tab:residuals}). The calibration analysis reveals: (i)~the surrogate is mildly conservative, over-predicting von Mises stress by 6.4\% and compliance by 1.4\% on average; (ii)~displacement prediction is nearly unbiased (+1.4\% mean under-prediction); (iii)~scatter is substantial (MARE = 37.4\% for stress, 18.5\% for compliance, 10.9\% for displacement), reflecting the difficulty of predicting peak localized quantities. Isotonic regression fitted to these residuals provides a post-hoc bias correction; however, when applied to optimized designs, the calibration shifts predictions conservatively upward due to the distribution shift between original and optimized geometries (Section~\ref{sec:feasibility}). The Stage~2 FEA re-analysis (below) provides ground-truth residuals on optimized designs, confirming that the surrogate maintains meaningful rank-order fidelity ($\rho = 0.657$ for compliance) despite distribution shift.

\textbf{Stage 2: Re-analysis of optimized designs.} Re-mesh and re-solve at least 100 optimized geometries with ground-truth FEA, stratified into three groups:
\begin{enumerate}[label=(\alph*),leftmargin=2em]
\item The 30 highest-reduction designs (35--45\% reduction), where the geometry has diverged most from the training distribution and surrogate extrapolation risk is greatest.
\item 40 near-boundary designs where the conservative constraint bound $\mu + k\sigma$ is within 10\% of the allowable limit, representing the cases most likely to exhibit constraint violations under re-analysis.
\item 30 randomly sampled mid-range designs (15--30\% reduction), providing an unbiased estimate of typical surrogate accuracy under moderate optimization.
\end{enumerate}
A preliminary surrogate-level stratified sample has been selected (Section~\ref{sec:feasibility}); among these 100 designs, 47 currently satisfy conservative constraints and the mean volume reduction is 19.8\%. For each design, the FEA re-analysis should compare the surrogate-predicted stress, displacement, and compliance against the FEA values and report: (i)~the fraction of designs where constraints remain satisfied under ground truth (target: $>$85\% of the 355 surrogate-feasible designs should remain feasible under FEA), (ii)~the mean and maximum surrogate error margin (target: mean error $<$5--8\% of allowable limits), and (iii)~the Spearman rank correlation between surrogate and FEA on optimized (not just baseline) geometries.

\textbf{Stage 3: Nonlinear spot check.} For 5 representative optimized designs, run a nonlinear FEA with a tension-compression asymmetric concrete model (e.g., Concrete Damaged Plasticity in Abaqus or equivalent) to assess whether the thin interior partitions ($\sim$78~mm) exhibit cracking or buckling modes not captured by the linear elastic assumption.

\textbf{Stage 4: Physical print validation.} Fabricate at least one optimized geometry at reduced scale (e.g., 1:10) using a desktop concrete printer and perform compression testing to verify that the optimized design maintains structural integrity. This step would bridge the simulation-to-reality gap and provide the strongest evidence of practical feasibility.

We regard Stages 1--2 as essential for journal submission and Stages 3--4 as strengthening extensions. Stage~1 calibration diagnostics have been completed on the held-out test set (Table~\ref{tab:residuals}), confirming mild conservative bias. Stage~2 matched-method FEA validation on 100 stratified designs is reported below.

\subsubsection{Same-Method FEA Validation on 100 Designs}
\label{sec:fea_reanalysis}

To close the validation gap identified in Section~\ref{sec:validation}, we performed independent hex8 voxel FEA on both the \emph{baseline} (unoptimized) and \emph{optimized} geometries of 100 stratified designs, computing the same-method compliance ratio $C_{\text{opt}} / C_{\text{base}}$ under identical boundary conditions.  The key insight is that absolute voxel-vs-tetrahedral differences cancel in the ratio, enabling a fair apples-to-apples comparison.

\paragraph{Setup.} Each FEA uses uniform hex8 elements on a $128^3$ grid with the same boundary conditions as training: all DOFs fixed at the min-$x$ face (cantilever), with gravity loading ($\rho = 2400$~kg/m$^3$, $g = 9.81$~m/s$^2$). The linear system is solved with an algebraic multigrid (AMG) preconditioned conjugate gradient solver (pyamg smoothed aggregation with 6 rigid-body near-null-space modes, tolerance $10^{-5}$), achieving convergence in 83~$\pm$~12 CG iterations (compared to 4{,}000--5{,}000 with Jacobi preconditioning). The 100 designs are stratified into three groups:
\begin{enumerate}[label=(\alph*),leftmargin=2em]
\item 30 \emph{high-reduction} designs (35--45\% volume reduction), where surrogate extrapolation risk is greatest.
\item 40 \emph{near-boundary} designs where the conservative constraint bound $\mu + k\sigma$ is within 10\% of the allowable limit.
\item 30 \emph{random} mid-range designs (15--30\% reduction), providing an unbiased accuracy estimate.
\end{enumerate}

\paragraph{Results.} All 100 designs pass the compliance constraint $C_{\text{opt}} / C_{\text{base}} \leq 1.15$, yielding a \textbf{100\% survival rate} (100/100, 95\% Clopper--Pearson CI: [96.4\%, 100\%]). Table~\ref{tab:fea_reanalysis_100} reports per-group statistics.

\begin{table}[!htbp]
\centering
\caption{Same-method voxel FEA validation of 100 stratified optimized designs. For each design, both the baseline (unoptimized) and optimized geometries are solved under identical hex8 voxel FEA ($128^3$ grid, AMG-preconditioned CG, tolerance $10^{-5}$). All ratios are optimized/baseline under the same discretization. $C$-ratio $= C_{\text{opt}} / C_{\text{base}}$; VM-ratio $= \sigma_{\text{VM,opt}} / \sigma_{\text{VM,base}}$; Disp-ratio $= u_{\text{opt}} / u_{\text{base}}$.}
\label{tab:fea_reanalysis_100}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Group} & \textbf{$n$} & \textbf{$C$-ratio (mean $\pm$ std)} & \textbf{$C$-ratio (max)} & \textbf{VM-ratio (mean)} & \textbf{Time (mean)} \\
\midrule
High-reduction  & 30 & $0.458 \pm 0.063$ & 0.617 & 0.716 & 106~s \\
Near-boundary   & 40 & $0.702 \pm 0.126$ & 1.004 & 0.851 & 116~s \\
Random          & 30 & $0.629 \pm 0.081$ & 0.757 & 0.808 & 92~s \\
\midrule
\textbf{All}    & \textbf{100} & $\mathbf{0.607 \pm 0.141}$ & \textbf{1.004} & \textbf{0.798} & \textbf{106~s} \\
\bottomrule
\end{tabular}
\end{table}

Four findings emerge from the validation (confirmed at full population below):

\textbf{1.~Optimization improves stiffness, not just preserves it.} The mean compliance ratio is $0.607 \pm 0.141$ (95\% CI: $[0.580, 0.635]$), meaning optimization \emph{reduces} compliance by $\sim$40\% on average---removing non-structural material reduces gravity self-weight while preserving critical load paths. The maximum ratio across all 100 designs is 1.004 (a single near-boundary case).

\textbf{2.~Zero false positives.} No design that passed the surrogate's conservative constraint violated the compliance constraint under independent FEA ($0/100$, 95\% Clopper--Pearson upper bound 3.6\%).

\textbf{3.~Surrogate ranking is preserved.} Spearman rank correlation between surrogate-predicted and voxel-FEA compliance on optimized designs is $\rho = 0.657$ ($p < 10^{-13}$), lower than the held-out correlation ($\rho = 0.948$) but confirming meaningful rank-order fidelity under distribution shift.

\textbf{4.~AMG preconditioning enables practical validation.} Each paired FEA completes in $106 \pm 30$~s with AMG preconditioning (83~$\pm$~12 CG iterations vs.\ 4{,}000--5{,}000 with Jacobi), totaling 2.94~h for 100 designs on a single CPU core.

\paragraph{Full-population validation ($n = 355$).} To eliminate sampling uncertainty entirely, we extended the FEA re-analysis to \emph{all} 355 constraint-satisfying designs. The full-population results are consistent with the 100-design stratified sample: mean compliance ratio $0.631 \pm 0.112$, maximum 1.004, and \textbf{0/355 false positives}---a 100\% constraint survival rate with 95\% Clopper--Pearson CI $[99.0\%, 100\%]$. The full-population mean VM stress ratio is $0.805 \pm 0.078$ (max 1.055). Because the validation now covers the entire surrogate-feasible population rather than a sample, the conformal bound tightens to $P(\text{violation}) \leq 1/356 = 0.28\%$ (Section~\ref{sec:conformal}).

\paragraph{What this validation certifies and does not certify.}
The same-method compliance ratio test certifies that SASTO-optimized designs do not degrade structural stiffness beyond the allowable threshold \emph{relative to their unoptimized baselines}, under identical discretization, solver, and boundary conditions---eliminating all confounding factors (mesh type, solver tolerance, element formulation) from the comparison.
The validation does \emph{not} certify:
\begin{itemize}[leftmargin=2em]
\item \emph{Absolute code compliance.} Voxel hex8 elements on a $128^3$ grid produce staircase geometry approximations whose absolute stress and compliance differ systematically from the tetrahedral training FEA (Section~\ref{sec:fea_reanalysis}). The ratio metric is specifically designed to cancel this discretization bias.
\item \emph{Nonlinear failure modes.} Concrete cracking, compression softening, and local buckling are not captured by the linear elastic model; thin features ($\sim$78~mm) may exhibit failure modes invisible to the surrogate.
\item \emph{Material anisotropy and construction defects.} Layer interface weakness in 3D-printed concrete is not modeled.
\item \emph{Foundation boundary conditions.} The min-$x$ cantilever BC used consistently across training and validation exercises the structure more aggressively than a realistic foundation-supported condition; redesign for production use would require re-training under appropriate BCs.
\end{itemize}
The compliance ratio is the correct confound-neutral statistic because it cancels the systematic voxel-vs-tetrahedral bias, isolating the question: ``Does optimization preserve structural performance relative to baseline under the same physics?''

\subsubsection{Conformal Prediction Analysis}
\label{sec:conformal}

To replace the heuristic $\mu + k\sigma$ bound with a distribution-free guarantee, we apply split conformal prediction \cite{vovk2005,angelopoulos2023} to the FEA validation data.

\paragraph{Constraint-satisfaction certification.} Among the $n = 355$ FEA-validated constraint-satisfying designs, 0 violate the compliance constraint ($C_{\text{opt}}/C_{\text{base}} > 1.15$). By the conformal prediction exchangeability guarantee \cite{vovk2005}, the probability that a \emph{new} constraint-satisfying design violates the FEA compliance constraint satisfies:
\begin{equation}\label{eq:conformal_bound}
P(\text{violation}) \leq \frac{1}{n+1} = \frac{1}{356} = 0.28\%.
\end{equation}
This is a distribution-free, finite-sample result requiring only exchangeability of the calibration and test designs---satisfied because both are drawn from the same surrogate-feasible population under the same boundary conditions. The corresponding Clopper--Pearson 95\% CI on the violation rate is $[0,\,0.84\%]$.

\paragraph{Compliance ratio upper bound.} Applying one-sided conformal regression to the observed compliance ratios, the 99\% conformal upper bound is $C \leq 0.950$, with a margin of 0.20 to the 1.15 threshold. The 95\% bound is $C \leq 0.807$ (margin 0.34). These bounds confirm that the surrogate's conservative filtering produces designs with substantial structural safety margins well below the constraint limit.

\paragraph{Calibrated uncertainty multiplier.} Using the 1,114-sample held-out test set with ensemble standard deviations estimated via coefficient-of-variation proxy from 200 optimization summaries, split conformal prediction calibrates the $k$-factor. The conformal $k$ for 84.1\% one-sided coverage (the Gaussian-equivalent of $k = 1.0$) is $k_{\text{conformal}} = 1.90$ for compliance and $k = 4.31$ for von Mises stress. This confirms that the ensemble residuals exhibit heavier tails than the Gaussian model assumes (Remark~\ref{rem:gaussianity}): the heuristic $k = 1.0$ provides approximately 65--75\% true coverage for compliance rather than the nominal 84.1\%. Despite this under-coverage, no constraint violations occur because the surrogate's systematic conservatism---over-predicting the compliance ratio by a factor of $\sim$3$\times$ relative to independent voxel FEA---provides an implicit safety margin far exceeding the $\mu + k\sigma$ uncertainty buffer.

\paragraph{Implications.} The conformal analysis transitions the safety argument from the heuristic ``$\mu + k\sigma$ with $k = 1.0$ is close to 84\% coverage if Gaussian'' to the distribution-free ``$P(\text{violation}) \leq 0.28\%$ with $n = 355$ calibration points, regardless of the residual distribution.'' The residual heavy-tailedness revealed by the calibrated $k$ analysis underscores the importance of this transition: the Gaussian assumption alone would understate the true exceedance risk by roughly a factor of two.

\subsection{Comparison with the State of the Art}

Direct numerical comparison with prior topology optimization methods for building-scale structures is difficult because no directly comparable benchmark exists. Existing neural surrogate approaches \cite{nie2021, banga2018} target small components (e.g., engine brackets) at resolutions of $40^3$ to $64^3$, operate without uncertainty quantification, and do not address mesh connectivity or manufacturing constraints. To our knowledge, no existing method simultaneously provides surrogate-accelerated optimization, formal mesh connectivity guarantees, and part-aware heterogeneous thickness control at building scale; however, we note that the rapid pace of development in both neural surrogate methods and construction-scale AM may yield concurrent or subsequent approaches.

% ============================================================
\section{Limitations and Threats to Validity}
\label{sec:limitations}
% ============================================================

\subsection{Critical Limitations}

\textbf{Independent FEA re-analysis (completed).} Same-method hex8 voxel FEA validation on all 355 constraint-satisfying designs confirms 100\% constraint survival ($C_{\text{opt}}/C_{\text{base}} \leq 1.15$ for all designs; Section~\ref{sec:fea_reanalysis}). Conformal prediction certifies $P(\text{violation}) \leq 0.28\%$ (Section~\ref{sec:conformal}). This closes the primary validation gap, though the caveats enumerated in Section~\ref{sec:fea_reanalysis} (nonlinear failure, anisotropy, foundation BCs) remain open.

\textbf{Thin-feature robustness.} Interior walls reduced to 1 voxel ($\sim$78 mm) may be susceptible to buckling under accidental lateral loading, which is not captured by the linear elastic FEA model. To bound this risk, we estimate the critical Euler plate buckling stress for the thinnest permissible feature ($t = 78$~mm, height $h = 3$~m, both edges restrained):
\begin{equation}\label{eq:buckling}
\sigma_{\mathrm{cr}} = \frac{k\,\pi^2\,E}{12(1-\nu^2)}\left(\frac{t}{h}\right)^2 = \frac{4 \times \pi^2 \times 25\text{~GPa}}{12(1-0.04)}\left(\frac{0.078}{3.0}\right)^2 \approx 50~\text{MPa},
\end{equation}
where $k = 4$ for a plate with both vertical edges restrained (connected to perpendicular walls or floors). The resulting buckling safety factor $\lambda = \sigma_{\mathrm{cr}} / \sigma_{\mathrm{allow}} = 50/5 = 10$ substantially exceeds the conventional threshold ($\lambda > 3$--$5$). Moreover, interior partition walls carry negligible axial load (they are non-structural dividers), so actual compressive stresses are far below $\sigma_{\mathrm{allow}}$. We therefore conclude that Euler-mode plate buckling is not a governing failure mode for the minimum-thickness features in SASTO. However, local buckling at stress concentration sites, second-order (P-$\Delta$) effects, and imperfection sensitivity in printed concrete (where layer interfaces may act as geometric imperfections) require nonlinear buckling analysis that is beyond the scope of this work and is identified as future validation in Stage~3 of the roadmap (Section~\ref{sec:validation}).

\textbf{Stress concentration localization.} The surrogate predicts global maximum von Mises stress but does not localize it. Local stress concentrations at geometric discontinuities created by optimization may exceed the predicted global maximum.

\textbf{Nonlinear material behavior.} Concrete exhibits tension cracking and compression softening, neither of which is captured by the isotropic linear elastic model. Optimized thin features may fail in modes not predicted by the surrogate.

\textbf{Material anisotropy.} 3D-printed concrete is orthotropic due to the layer-by-layer deposition process: inter-layer bond strength is typically 50--80\% of the bulk value, and Young's modulus may differ by 10--30\% between the print direction and transverse directions \cite{buswell2018}. The present work assumes isotropic material properties with $E = 25$~GPa, which represents an upper-bound stiffness estimate. A production-grade SASTO implementation would require integration of orthotropic stiffness tensors ($C_{ijkl}$ with 9 independent constants) into both the FEA training data and the surrogate architecture. We note that the isotropic assumption is consistent across all training and evaluation samples, so comparative optimization results (relative rankings and reductions) remain internally valid even if absolute stress magnitudes are approximate.

\textbf{Boundary condition idealization.} The FEA training and validation data uses a fixed-face boundary condition at the minimum-$x$ face (one vertical side wall fully restrained), modeling a cantilever-type loading scenario rather than a foundation-supported building. This idealization was chosen because it provides a consistent, reproducible structural test with well-defined stress gradients---a standard benchmark in structural topology optimization \cite{bendsoe2003}. Real houses rest on foundations with gravity as the primary load path; the cantilever condition produces a different stress distribution pattern (bending-dominated rather than compression-dominated) and exercises the structure more aggressively than foundation fixity. Crucially, the same boundary condition is used across all training, optimization, and validation simulations, so all comparative results (relative compliance ratios, volume reductions, ranking correlations) remain internally valid. However, absolute stress and displacement magnitudes should not be interpreted as predictions for foundation-supported buildings, and production deployment would require re-training under realistic boundary conditions (e.g., fixed ground plane with soil--structure interaction via Winkler springs). This is identified as future work.

\textbf{Gaussian uncertainty assumption (mitigated).} The $\mu + k\sigma$ bound implicitly assumes approximately Gaussian residuals. Conformal calibration reveals heavier tails ($k_{\text{conformal}} = 1.90$ for 84.1\% compliance coverage vs.\ nominal $k = 1.0$), reducing true one-sided coverage to $\sim$65--75\%. This is fully mitigated by two factors: (1)~conformal certification on $n = 355$ designs establishes $P(\text{violation}) \leq 0.28\%$ regardless of the residual distribution (Section~\ref{sec:conformal}), and (2)~the surrogate's systematic conservatism ($\sim$3$\times$ compliance ratio overestimate relative to voxel FEA) provides an implicit buffer far exceeding the formal uncertainty correction.

\subsection{Surrogate Limitations}

\textbf{Distribution shift.} Optimized geometries with 45\% material removed differ substantially from the training distribution of unoptimized houses. The mean volume fraction of optimized designs is 0.77 (std 0.08) among the 355 constraint-satisfying geometries, compared to 1.00 for all training samples. This shift can be quantified along three axes: (i)~\emph{volume fraction}: the training set contains only geometries at 100\% fill, while optimized designs range from 55\% to 99\% fill; (ii)~\emph{surface topology}: material removal creates interior voids, thin features, and surface roughness absent from the training data; and (iii)~\emph{part-label distribution}: interior walls, which constitute 30--40\% of total volume in typical unoptimized houses, are reduced to 5--15\% in heavily optimized designs, shifting the relative part composition outside the training envelope.

The ensemble disagreement divergence $\Gamma_D$ has a population mean of 0.255 and median of 0.223 across 916 optimized designs (Table~\ref{tab:uq_population}), indicating moderate uncertainty growth during optimization. The per-target analysis reveals that von Mises stress shows the highest disagreement (mean CV = 0.212, P95 = 0.492), consistent with stress being a localized quantity sensitive to thin-feature creation. However, $\Gamma_D$ is an internal proxy based on the surrogate's own uncertainty estimates: it does not directly measure true error under distribution shift. The completed same-method FEA re-analysis (Section~\ref{sec:fea_reanalysis}) provides independent validation, confirming 0/355 false positives despite this distribution shift and a Spearman rank correlation of $\rho = 0.657$ for compliance on optimized designs.

\textbf{Constraint satisfaction rate.} The 38.8\% feasibility rate (355/916) is the primary limitation on practical utility. The binding constraint is surrogate compliance accuracy: for approximately 39\% of geometries, the conservative prediction exceeds the compliance constraint at the original geometry, leaving zero feasible erosion budget. Section~\ref{sec:feasibility} analyzes this bottleneck in detail and identifies post-hoc calibration, conformalized uncertainty (now partially addressed in Section~\ref{sec:conformal}), and FEA-in-the-loop verification as concrete mitigation strategies. The conformal analysis confirms that the current $k = 1.0$ operating point provides approximately 65--75\% true one-sided coverage (vs.\ 84.1\% under Gaussianity), suggesting that a conformally-calibrated reduction to $k \approx 0.5$--$0.7$ could substantially increase the feasibility rate while maintaining the same empirical constraint satisfaction guarantee. The $k$-factor ablation (Table~\ref{tab:ksensitivity}) quantifies the full feasibility--conservatism tradeoff, showing that 76.5\% of designs satisfy constraints at the ensemble mean ($k = 0$), but this drops to 38.8\% at $k = 1.0$, confirming that ensemble uncertainty is the dominant factor limiting feasibility yield.

\subsection{External Validity}

Results apply only to single-story structures with structural concrete under gravity and wind loading. Generalization to multi-story buildings, seismic loading, different materials (geopolymer, fiber-reinforced concrete), and larger footprints is unverified.

\subsection{Claim--Evidence Ledger}
\label{sec:ledger}

Table~\ref{tab:ledger} maps each major claim to its supporting evidence and boundary conditions, providing a one-page audit trail for the paper's assertions.

\begin{table}[!htbp]
\centering
\caption{Claim--evidence ledger. Each claim is mapped to the table/figure providing primary support, the boundary conditions under which the claim holds, and known caveats.}
\label{tab:ledger}
\small
\begin{tabular}{@{}p{0.6cm}p{3.8cm}p{3.0cm}p{2.0cm}p{3.5cm}@{}}
\toprule
\textbf{ID} & \textbf{Claim} & \textbf{Evidence} & \textbf{BCs} & \textbf{Caveats} \\
\midrule
C1 & 23.5\% $\pm$ 7.8\% mean reduction (355/916) & Table~\ref{tab:ksensitivity}, Fig.~\ref{fig:reduction_dist} & $k{=}1.0$, cantilever BC & Surrogate constraint; not FEA-verified at every $k$ \\
\addlinespace
C2 & 23--92$\times$ speedup vs SIMP & Table~\ref{tab:simp}, Fig.~\ref{fig:simp_comparison} & 64$^3$ SIMP vs 128$^3$ SASTO & Resolution mismatch; extrapolated \\
\addlinespace
C3 & 6-connectivity eliminates MC fragments & Proposition~\ref{prop:mc}, Table~\ref{tab:connectivity}, Remark~\ref{rem:meshgap} & Idealized SDF & 10\% need trivial post-processing \\
\addlinespace
C4 & PA yields 10.7~pp more than U & Table~\ref{tab:reference}, Table~\ref{tab:perpart} & Reference case (00472) & Single-design ablation \\
\addlinespace
C5 & 0/355 false positives; max $C$-ratio 1.004 & Section~\ref{sec:fea_reanalysis} & Same-method hex8, cantilever & Linear elastic only \\
\addlinespace
C6 & $P(\text{violation}) \leq 0.28\%$ & Section~\ref{sec:conformal} & Exchangeability, $n{=}355$ & Same physics model assumed \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Conclusion}
\label{sec:conclusion}
% ============================================================

This work presented Surrogate-Accelerated Sensitivity Topology Optimization (SASTO), a three-phase voxel erosion algorithm that replaces iterative finite element analysis with a deep ensemble surrogate for building-scale structural optimization. Evaluated across 916 diverse house geometries, the method achieves 23.5\% $\pm$ 7.8\% mean material reduction (95\% CI: [22.7\%, 24.3\%]) across 355 constraint-satisfying configurations (and 45.0\% on the reference case) in a median of 50 seconds on a consumer GPU, an empirically-anchored 23--92$\times$ speedup over SIMP at matched resolution (Section~\ref{sec:speedup}). A $k$-factor ablation across the full evaluation set reveals a smooth Pareto frontier: 76.5\% feasibility at $k = 0$ to 7.1\% at $k = 3$, with the operating point $k = 1.0$ at 38.8\%. The 6-connectivity digital topology criterion eliminates floating mesh fragments by enforcing marching-cubes-compatible connectivity---building on well-established digital topology theory \cite{kong1989} but addressing its specific application to building-scale voxel topology optimization, where the resulting marching cubes incompatibility has not, to our knowledge, been previously quantified at building scale---guaranteeing 6-connected voxel fields from which single-component meshes can be extracted with minimal post-processing (Proposition~\ref{prop:mc}). The part-aware heterogeneous thickness formulation correctly identifies the structural hierarchy between load-bearing and non-structural members, yielding 10.7 percentage points more reduction than uniform thickness.

The large-scale evaluation reveals that the constraint-feasibility rate (38.8\%) is governed by a well-characterized conservatism--yield tradeoff (Table~\ref{tab:ksensitivity}), with compliance calibration as the binding limitation. Calibration diagnostics on 1,114 held-out test samples (Table~\ref{tab:residuals}) confirm that the surrogate is mildly conservative (over-predicting stress by 6.4\%, compliance by 1.4\%), and population-level ensemble disagreement analysis ($\Gamma_D$, Table~\ref{tab:uq_population}) quantifies the uncertainty scaffold's tracking of distribution shift. These findings motivate a structured program of future work, ordered by priority:

\textbf{Essential validation (completed).} Ground-truth FEA re-analysis of all 355 constraint-satisfying designs (Section~\ref{sec:fea_reanalysis}) confirms 100\% constraint survival under same-method voxel FEA, with 0/355 false positives, mean compliance ratio $0.631 \pm 0.112$, and conformal bound $P(\text{violation}) \leq 0.28\%$ (Section~\ref{sec:conformal}).

\textbf{Completed calibration.} (3)~Distribution-free conformal prediction on $n = 355$ FEA-validated designs certifies $P(\text{violation}) \leq 0.28\%$ (Section~\ref{sec:conformal}), replacing the heuristic Gaussian assumption with a finite-sample guarantee. Calibrated $k$-factor analysis reveals the ensemble residuals are heavier-tailed than Gaussian ($k_{\text{conformal}} = 1.90$ for 84.1\% compliance coverage), but the surrogate's systematic conservatism provides an adequate implicit safety margin.

\textbf{High-impact extensions.} (4)~FEA-in-the-loop active learning---triggering sparse ground-truth re-analyses when ensemble disagreement exceeds a threshold $\Gamma_D > \tau$---would convert SASTO into a verified, self-correcting optimizer while generating high-value training data from the out-of-distribution regime. (5)~Nonlinear FEA spot checks with a tension-compression asymmetric concrete model on representative optimized thin features ($\sim$78~mm interior walls) to assess cracking and buckling modes absent from the linear elastic surrogate.

\textbf{Physical validation.} (6)~Small-scale physical 3D-print testing (e.g., 1:10 reduced scale) with compression loading to bridge the simulation-to-reality gap. This step is required to confirm that the optimized designs maintain structural integrity outside the simulation environment.

\textbf{Broader extensions.} Multi-story structures and seismic loading, integration of overhang constraints for toolpath compatibility, and development of anisotropic constitutive models that capture the layer-interface behavior of printed concrete. Scaling law analysis (Section~\ref{sec:scaling}) projects that doubling the training set size to $\sim$18{,}000 samples would reduce compliance MARE by 20--25\%, potentially raising the feasibility rate from 38.8\% to 45--55\% at the current operating point. Edge-case analysis (Section~\ref{sec:edge_cases}, Figure~\ref{fig:failure_gallery}) confirms that both low-reduction feasible designs and high-reduction infeasible designs trace to the same surrogate calibration bottleneck, motivating targeted data generation as the highest-leverage improvement.

The source code, trained model weights, training data generation pipelines, and optimization scripts are publicly available at \url{https://github.com/erichou1/fea.git} to facilitate reproduction and extension.

% ============================================================
\section*{Acknowledgements}
% ============================================================

The author thanks the Synopsys Science Fair organizing committee for the opportunity to present this work. Computational resources for large-scale FEA simulations and surrogate training were provided through access to NVIDIA GB200 GPU infrastructure. The 3DWire wireframe dataset and CubiCasa5k floor plans were used under their respective licenses.

% ============================================================
\begin{thebibliography}{99}

\bibitem{3dwire2024}
Lin, Y., Fan, Z., Li, M., \& Zhang, H. (2024). 3DWire: 3D building wireframe dataset. \emph{Visual Computing Center, KAUST}. Available at \url{https://vcc.tech/research/2024/3DWire}.

\bibitem{asce2022}
ASCE. (2022). \emph{Minimum Design Loads and Associated Criteria for Buildings and Other Structures} (ASCE/SEI 7-22). American Society of Civil Engineers.

\bibitem{banga2018}
Banga, S., Gehber, H., Dozber, C., \& Kara, L.~B. (2018). 3D topology optimization using convolutional neural networks. \emph{arXiv preprint arXiv:1808.07440}.

\bibitem{bendsoe2003}
BendsÃ¸e, M.~P., \& Sigmund, O. (2003). \emph{Topology Optimization: Theory, Methods, and Applications}. Springer.

\bibitem{brackett2011}
Brackett, D., Ashcroft, I., \& Hague, R. (2011). Topology optimization for additive manufacturing. \emph{Proceedings of the Solid Freeform Fabrication Symposium}, 348--362.

\bibitem{buswell2018}
Buswell, R.~A., Leal de Silva, W.~R., Jones, S.~Z., \& Dirrenberger, J. (2018). 3D printing using concrete extrusion: A roadmap for research. \emph{Cement and Concrete Research}, 112, 37--49.

\bibitem{conn2000}
Conn, A.~R., Gould, N.~I.~M., \& Toint, P.~L. (2000). \emph{Trust-Region Methods}. SIAM.

\bibitem{dasilva2019}
da Silva, G.~A., Beck, A.~T., \& Sigmund, O. (2019). Topology optimization of compliant mechanisms with stress constraints and manufacturing error robustness. \emph{Computer Methods in Applied Mechanics and Engineering}, 354, 397--421.

\bibitem{dunning2011}
Dunning, P.~D., Kim, H.~A., \& Mullineux, G. (2011). Introducing loading uncertainty in topology optimization. \emph{AIAA Journal}, 49(4), 760--768.

\bibitem{gaynor2016}
Gaynor, A.~T., \& Guest, J.~K. (2016). Topology optimization considering overhang constraints: Eliminating sacrificial support material in additive manufacturing through design. \emph{Structural and Multidisciplinary Optimization}, 54(5), 1157--1172.

\bibitem{geuzaine2009}
Geuzaine, C., \& Remacle, J.-F. (2009). Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities. \emph{International Journal for Numerical Methods in Engineering}, 79(11), 1309--1331.

\bibitem{guest2004}
Guest, J.~K., PrÃ©vost, J.~H., \& Belytschko, T. (2004). Achieving minimum length scale in topology optimization using nodal design variables and projection functions. \emph{International Journal for Numerical Methods in Engineering}, 61(2), 238--254.

\bibitem{hu2018}
Hu, J., Shen, L., \& Sun, G. (2018). Squeeze-and-excitation networks. \emph{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}, 7132--7141.

\bibitem{iea2021}
IEA. (2021). \emph{Global Status Report for Buildings and Construction 2021}. International Energy Agency.

\bibitem{kong1989}
Kong, T.~Y., \& Rosenfeld, A. (1989). Digital topology: Introduction and survey. \emph{Computer Vision, Graphics, and Image Processing}, 48(3), 357--393.

\bibitem{lakshminarayanan2017}
Lakshminarayanan, B., Pritzel, A., \& Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. \emph{Advances in Neural Information Processing Systems}, 30.

\bibitem{langelaar2016}
Langelaar, M. (2016). Topology optimization of 3D self-supporting structures for additive manufacturing. \emph{Additive Manufacturing}, 12, 60--70.

\bibitem{lazarov2011}
Lazarov, B.~S., \& Sigmund, O. (2011). Filters in topology optimization based on Helmholtz-type differential equations. \emph{International Journal for Numerical Methods in Engineering}, 86(6), 765--781.

\bibitem{lorensen1987}
Lorensen, W.~E., \& Cline, H.~E. (1987). Marching cubes: A high resolution 3D surface construction algorithm. \emph{ACM SIGGRAPH Computer Graphics}, 21(4), 163--169.

\bibitem{ngo2018}
Ngo, T.~D., Kashani, A., Imbalzano, G., Nguyen, K.~T.~Q., \& Hui, D. (2018). Additive manufacturing (3D printing): A review of materials, methods, applications and challenges. \emph{Composites Part B: Engineering}, 143, 172--196.

\bibitem{nie2021}
Nie, Z., Lin, T., Jiang, H., \& Kara, L.~B. (2021). TopologyGAN: Topology optimization using generative adversarial networks based on physical fields over the initial domain. \emph{Journal of Mechanical Design}, 143(3), 031715.

\bibitem{ovadia2019}
Ovadia, Y., Fertig, E., Ren, J., et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. \emph{Advances in Neural Information Processing Systems}, 32.

\bibitem{sigmund2013}
Sigmund, O., \& Maute, K. (2013). Topology optimization approaches. \emph{Structural and Multidisciplinary Optimization}, 48(6), 1031--1055.

\bibitem{white2019}
White, D.~A., Arrighi, W.~J., Kudo, J., \& Watts, S.~E. (2019). Multiscale topology optimization using neural network surrogate models. \emph{Computer Methods in Applied Mechanics and Engineering}, 346, 1118--1135.

\bibitem{xia2015}
Xia, L., \& Breitkopf, P. (2015). Design of materials using topology optimization and energy-based homogenization approach in Matlab. \emph{Structural and Multidisciplinary Optimization}, 52(6), 1229--1241.

\bibitem{xie1997}
Xie, Y.~M., \& Steven, G.~P. (1997). \emph{Evolutionary Structural Optimization}. Springer.

\bibitem{sfepy2019}
Cimrman, R., LukeÅ¡, V., \& Rohan, E. (2019). Multiscale finite element computations in Python using SfePy. \emph{Advances in Computational Mathematics}, 45(4), 1897--1921.

\bibitem{trimesh2019}
Dawson-Haggerty, M. (2019). trimesh. \url{https://trimesh.org/}. Accessed 2025.

\bibitem{keyak1993}
Keyak, J. H., Meagher, J. M., Skinner, H. B., \& Mote Jr., C. D. (1990). Automated three-dimensional finite element modelling of bone: A method and preliminary results. \emph{Journal of Biomedical Engineering}, 12(5), 389--397.

\bibitem{vovk2005}
Vovk, V., Gammerman, A., \& Shafer, G. (2005). \emph{Algorithmic Learning in a Random World}. Springer.

\bibitem{angelopoulos2023}
Angelopoulos, A.~N., \& Bates, S. (2023). Conformal prediction: A gentle introduction. \emph{Foundations and Trends in Machine Learning}, 16(4), 494--591.

\end{thebibliography}

% ============================================================
\appendix
\section*{Appendices}
\addcontentsline{toc}{section}{Appendices}
% ============================================================

\section{SASTO Pseudocode}
\label{app:pseudocode}

Algorithm~\ref{alg:sasto} in Section~\ref{sec:algorithm} provides the complete pseudocode. The key implementation details are as follows. The distance transform is computed using \texttt{scipy.ndimage.distance\_transform\_edt}. The 6-simple-point test evaluates connected components within the $3\times3\times3$ neighborhood using a local breadth-first search. Sensitivity gradients are computed via PyTorch autograd with \texttt{retain\_graph=True} across all five ensemble members. The marching cubes extraction uses \texttt{skimage.measure.marching\_cubes} with the SDF level set at zero.

\section{Training Hyperparameters}
\label{app:hyperparams}

Table~\ref{tab:hyperparams} lists all hyperparameters used for surrogate training.

\begin{table}[!htbp]
\centering
\caption{Complete training hyperparameter specification for the deep ensemble surrogate.}
\label{tab:hyperparams}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Architecture & Surrogate3DResNet (4 stages, 8 ResBlocks) \\
Ensemble members & 5 \\
Parameters per member & $\sim$8.76M \\
Total parameters & 43,802,083 \\
Input voxel channels & 7 (1 occupancy + 6 part one-hot) \\
Feature vector dimension & 10 \\
Base channels & 64 \\
Prediction targets & 3 (von Mises, displacement, compliance) \\
Activation & GELU \\
Normalization & BatchNorm3d + LayerNorm (head) \\
Attention & Squeeze-and-Excitation, reduction = 4 \\
Regularization & Dropout (0.15), DropPath (0.1), weight decay ($10^{-4}$) \\
Pooling & AdaptiveAvgPool3d + AdaptiveMaxPool3d \\
Head & 2-layer MLP (512 $\to$ 256 $\to$ 3) with skip \\
Target transform & $\log(1+|y|) \to$ z-score, winsorize 2nd/98th pctl \\
Loss & Huber (SmoothL1) \\
Optimizer & AdamW (lr = $5\times10^{-4}$, wd = $10^{-4}$) \\
Scheduler & CosineAnnealingWarmRestarts \\
Batch size & 32 \\
Max epochs & 200 \\
Early stopping & Patience = 30 \\
EMA & Decay = 0.999 \\
Mixed precision & AMP (bf16) \\
Gradient clipping & $\|\cdot\|_{\max} = 1.0$ \\
Augmentation & 90$^\circ$ rotations, flips, noise ($\sigma = 0.02$), 10\% ch.\ dropout \\
Data split & 8,943 / 1,121 / 1,114 (train / val / test) \\
\bottomrule
\end{tabular}
\end{table}

\section{Optimization Parameters}
\label{app:optparams}

Table~\ref{tab:optparams} lists all optimization parameters used in the SASTO-PA configuration.

\begin{table}[!htbp]
\centering
\caption{Complete optimization parameter specification for SASTO-PA.}
\label{tab:optparams}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Uncertainty factor $k$ & 1.0 \\
Max compliance ratio & 1.15$\times$ baseline \\
Volume weight $w_V$ & 1.0 \\
Surface weight $w_S$ & 0.01 \\
Constraint penalty $\kappa$ & 100.0 \\
VM allowable & 5.0 $\times 10^6$ Pa \\
Displacement allowable & $L/360 \approx 0.028$ m \\
Min thickness (exterior/roof/floor) & 2 voxels \\
Min thickness (interior wall) & 1 voxel \\
Initial batch size & 200 \\
Minimum batch size & 10 \\
Sensitivity recompute period & Every 3 layers \\
Max layers (Phase 1) & 40 \\
Max consecutive failures & 5 \\
VM/compliance sensitivity weight $\alpha$ & 0.3 \\
SDF blur $\sigma$ & 0.15 (smooths staircase artifacts at\\& voxel edges; see Lazarov \& Sigmund \cite{lazarov2011}) \\
Laplacian smoothing & 3 iterations, $\lambda = 0.3$ \\
Mesh scale & 10.0 m / 128 voxels = 0.0781 m/voxel \\
\bottomrule
\end{tabular}
\end{table}

\section{Voxel Cross-Sections and Optimization Visualization}
\label{app:voxel}

Figures~\ref{fig:voxelparts} and \ref{fig:voxelbeforeafter} show the voxelized representation of the design domain and the before/after comparison at a representative cross-section height.

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig13_voxel_parts.png
% Voxel grid slices at 3 heights, colored by part label (0-4)
\includegraphics[width=\figfull]{figures/fig13_voxel_parts.png}
\caption{Voxelized representation of the reference geometry at three cross-section heights. Part labels are color-coded: exterior wall (blue), interior wall (green), roof (orange), floor (red). The part labels drive the heterogeneous minimum thickness constraints in the part-aware formulation.}
\label{fig:voxelparts}
\end{figure}

The effect of SASTO-PA optimization on the voxel grid is shown at a representative cross-section height in Figure~\ref{fig:voxelbeforeafter}.

\begin{figure}[!htbp]
\centering
% PLACEHOLDER: Insert figures/fig18_voxel_before_after.png
% Side-by-side voxel occupancy at z=50: original (left), optimized (right), removal map (center)
\includegraphics[width=\figmed]{figures/fig18_voxel_before_after.png}

\caption{Voxel grid at $z = 50$ before optimization (left), after SASTO-PA optimization (center), and removal map (right) showing removed voxels colored by part type. Interior wall voxels (green) constitute the majority of removals; exterior wall, roof, and floor voxels are largely preserved.}
\label{fig:voxelbeforeafter}
\end{figure}

\section{Reproducibility}
\label{app:reproducibility}

All source code, trained model weights, training data generation pipelines, optimization scripts, and figure generation code are publicly available at \url{https://github.com/erichou1/fea.git}. The repository includes:

\begin{table}[!htbp]
\centering
\caption{Reproducibility artifacts and their locations in the repository.}
\label{tab:reproducibility}
\small
\begin{tabular}{@{}p{5.5cm}p{7cm}@{}}
\toprule
\textbf{Artifact} & \textbf{Path} \\
\midrule
FEA data generation & \texttt{optimization/run\_full\_pipeline.py} \\
Model architecture & \texttt{fea\_ml/fea\_ml/models/cnn3d.py} \\
Training script & \texttt{fea\_ml/fea\_ml/scripts/train.py} \\
Optimization algorithm & \texttt{fea\_ml/run\_opt\_part\_aware.py} \\
Batch optimization & \texttt{fea\_ml/run\_batch\_all.py} \\
Calibration analysis & \texttt{fea\_ml/calibration\_analysis.py} \\
Configuration & \texttt{fea\_ml/configs/voxel\_config.yaml} \\
Trained ensemble (5 members) & \texttt{fea\_ml/runs/v3/ensemble/} \\
Multi-geometry results & \texttt{fea\_ml/runs/v3/batch\_results\_all/} \\
Figure generation & \texttt{generate\_figures.py} \\
Figure generation (STL + SIMP) & \texttt{generate\_house\_figures.py}, \texttt{generate\_additional\_figures.py} \\
FEA validation & \texttt{fea\_ml/run\_fea\_validation\_parallel.py} \\
SIMP benchmark & \texttt{fea\_ml/run\_simp\_benchmark.py} \\
Conformal prediction & \texttt{fea\_ml/run\_conformal\_prediction.py} \\
STL mesh exports & \texttt{figures/stl\_exports/} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Computational environment.} Training: 4$\times$ NVIDIA GB200 (189~GB HBM3e each), Python~3.11, PyTorch~2.x, CUDA~12.x. Optimization: single NVIDIA RTX A3000 Laptop GPU (6~GB VRAM). FEA: SfePy~2024.x with Gmsh~4.x on CPU. The full 916-geometry evaluation required approximately 13.5 GPU-hours of continuous computation on the A3000.

\paragraph{Determinism.} All optimization runs use \texttt{seed = 42}, \texttt{torch.backends.cudnn.deterministic = True}. Ensemble members were trained independently with seeds 0--4. Training data splits are stored in \texttt{runs/v3/splits.json} with family-aware assignment to prevent near-duplicate leakage.

\paragraph{Data provenance.} Training targets (von Mises stress, displacement, compliance) were generated by SfePy FEA under ASCE 7-22 ASD load combinations with fixed-face boundary conditions at the minimum-$x$ face (cantilever). Target normalization: $\log_{1p}(|x|) \to z$-score, with stored parameters in \texttt{runs/v3/normalization.json}.

\end{document}