# Foundations of Operations Research

## Graph and Network optimization

### Representation

* Two nodes are **adjacent** if they are connected by an edge.
* An edge is **incident** in a node if the node is an endpoint for the edge.
* Two nodes are **connected** if there is a path connecting them.

**UNDIRECTED graph** with n nodes has $$m ≤ n(n-1)/2$$ edges. **DIRECTED graph** with n nodes has $$m ≤ n(n-1)$$ arcs.

A graph is **dense** (**adjacency matrix**) if $$m ≈ n^2$$, otherwise it is **sparse** if $$m << n^2$$ (**list of successors/predecessors**).

### Graph reachablity problem

#### BFS algorithm

BFS (breadth-first-search) algorithm uses a queue (FIFO) containing the nodes reachable from s and not yet processes. The algorithm stops when there are no more outgoing arcs from the subset M, $$δ^+(M) = ∅$$.

Input: G = (N, A) as successors list, and $$s ∈ N$$. \
Output: subset $$M ⊆ N$$ of nodes reachable from s.

```clike
Q <- {s}; 
M <- empty;
while Q != empty do
	//dequeue of the first node in Q and add it to M
	u <- node in Q;
	Q <- Q \ {u};
	M <- M U {u};
	//for each node reachable from u if not already present in M or Q
	for (u, v) in δ+(u) do
		if (v not in M AND v not in Q) then 
			Q <- Q U {v};
```

### Subgraphssrc="assets/ a connected and acyclic subgraph of G. It is called **spanning tree** of G if it contains all nodes of G.

Properties:

* Every tree with n nodes has n-1 edges.
* Any pair of nodes in a tree is connected via unique path.
* By adding a new edge to a tree, a unique cycle is created.
* Given a spanning tree T of G, consider and edge $$e ∉ T$$ and the unique cycle of $$T \cup {e}$$. For each edge $$f ∈ (C\setminus {e})$$ (of the cycle), the subgraph $$T \cup ({e} \setminus {f})$$ is also a spanning tree of G.
* Let F be a partial tree containing an optimal spanning tree of G. Consider $$e = {u, v} ∈ δ(S)$$ of minimum cost, then there exists a minimum cost spanning tree of G containing e.

### Minimum cost spanning trees

A complete graph with $$n ≥ 1$$ nodes has $$n^{(n-2)}$$ spanning trees.

#### Prim’s algorithm

Prim’s algorithm iteratively build a spanning tree. Start from initial tree $$TR = (S, T)$$ with $$S = {u}$$ and T empty. At each step, add to the current partial tree an edge of minimum cost among those which connect a node in S to a node in N\S.

Input: connected graph $$G = (N, A)$$ with edges costs. Output: subset $$T ⊆ N$$ of edges of G such that $$G’ = (N, T)$$ is a minimum cost spanning tree.

```c
S = {u};
T = emtpy;
while |T| < n-1 do
	{u, v} = edge in δ(S) of minimum cost with u ∈ S and v ∈ N\S.
	S = S U {v};
	T = T U {u, v}
```

It is an exact algorithm. It is greedy, as at each step a minimum cost edge is selected among all (locally optimal choice, without reconsidering previous choices).

*   Version with complexity $$O(n^2)$$

    ```c
    S = {u};
    T = emtpy;
    for j in N\S do
    	C_j = C_uj
    	closest_j = u;
    for k = 1,..,n-1 do
    	//select min edge in δ(S)
    	min = +inf;
    	for j=1,..,n do
    		if j not in S AND C_j < min then
    			min = C_j;
    			v = j;
    	S = S U {v};
    	T = T U {closest_v, v};
    	for j=1,..,n do
    		if j not in S AND c_vj < C_j then
    			C_j = c_vj;
    			closest_j = v;
    ```

### Kruskal’s algorithm

Create a forest F where each vertex is a separate tree. Create a set S containing all the edges in the graph. While S is not empty and F is not spanning: remove an edge with minimum cost from S, if the removed edge connects two different trees then add it to the forest F, combining two trees into a single tree.

```c
//sorted edges of E in non-decreasing order
S = {e1, ..., em};
F = empty;
i = 1;
while |F| < n-1 do
	if the two endpoints of e_i are in different trees then
		F = F U {e_i}
		merge the two trees containing the two endpoints
	i = i + 1;
```

### Optimality condition

A tree T is of **minimum total cost** if and only if **no cost-decreasing edge** exist. The optimality condition allows to verify if a spanning tree T is optimal: just check each $$e ∈ E\setminus T$$ is not a cost-decreasing edge.

### Shortest path problem

Given a directed graph $$G = (N, A)$$ with cost $$c_{ij} ∈ R$$ for each arc $$(i, j) ∈ A$$, and two nodes s (origin) and t (destination), determine a minimum cost (shortest) path from s to t.

If $$c_ij ≥ 0$$ for all $$(i, j) ∈ A$$, there is at least one shortest path which is simple. A path is simple if no node is visited more than once.

#### Dijkstra’s algorithm

Idea: consider the nodes in increasing order of length (cost) of the shortest path from s to any one of the other nodes. To each node $$j ∈ N$$ we assign two labels:

* $$L_j$$ which corresponds, at the end ot the algorithm, to the cost of a minimum cost path from s to j.
* $$pred_j$$ is associated with each node that represent the predecessor of j in a shortest path from s to j.

Input: $$G = (N, A)$$ with non-negative arc costs, $$s ∈ N$$. Output: shortest paths from s to all other nodes of G.

```c
//initialization
S = empty;
//X is a subset of nodes with temporary labels X = (N\S) ⊆ N
X = {s};
L_s = 0;
for u in N do
	L_u = inf;
while |S| != n do
	v = argmin{L_i: i in X};
	X = X \ {v};
	S = S U {v};
	//for each node reachable from v
	for (v, w) in δ+(v) do
		if L_w > L_v + c_vw then
			L_w = L_v + c_vw;
			pred_w = v;
			X = X U {v};
```

Dijkstra’s algorithm is an exact greedy algorithm that builds the shortest paths from a source node to all other nodes in the graph. A set of shortest paths from s to all other nodes can be retrieved backwards from t to s iterating over the predecessors.

#### Floyd-Warshall’s algorithm

It uses two $$n×n$$ matrices:

* D, where $$d_{ij}$$ corresponds to the cost of the shortest path from i to j.
* P, where $$p_{ij}$$ corresponds to the predecessor of j on the shortest path from i to j.

It is based on the **triangular operation**: for each pair of nodes i,j with $$i ≠ u$$ and $$j ≠ u$$ (including case $$i=j$$), check whether when going from i to j it is more convenient to go via u.

if $$d_{iu} + d_{uj} < d_{ij}$$ then $$d_{ij} = d_{iu} + d_{uj}$$

![triangular operation](</assets/Untitled (2).png>)

Input: a directed graph $$G = (N, A)$$ with an $$n×n$$ cost matrix $$C = [c_{ij}]$$. Output: for each pair of nodes $$i,j ∈ N$$, the cost $$c_{ij}$$ of the shortest path form i to j.

```c
//initialization of D and P
for i=1,..,n do
	for j=1,..,n do
		p_id = i;
		if i == j then 
				d_ij = 0;
		else if (i, j) in A then
				d_ij = c_ij;
		else 
				d_ij = inf;
//triangular operation for each node
for u in N do
	for i in N\{u} do
		for j in N\{u} do
			if d_iu + d_uj < d_ij then
				p_ij = p_uj;
				d_ij = d_iu + d_uj;
	for i in N do
			if d_ij < 0 then
				error "negative cycle"
```

It is an exact algorithm.

### Optimal paths in directed acyclic graphs

A **directed graph** is **acyclic** (DAG) if it contains no circuits. It is possible to define a **topological order**, that is an order given to the nodes so that for each arc $$(i, j) ∈ A$$ we have that the index of i is less tan the index of j.

<figure><img src="/assets/Untitled 1 (2).png" alt="" width="222"><figcaption></figcaption></figure>

Given a DAG $$G = (N, A)$$ represented via the lists of predecessors $$δ^−(v)$$ and successors $$δ^−(v)$$ for each node v. Assign the smallest positive integer not yet assigned to a node $$v ∈ N$$ with $$δ^−(v) = Ø$$. Delete the node v with all its incident arcs, then repeat until there are no left nodes.

Complexity: O(n + m) where n = |N| and m = |A| because each node or arc is considered at most once.

#### Dynamic programming algorithm for shortest paths in DAGs

Idea: any shortest path $$π_t$$ from 1 to t with at least 2 arcs, can be subdivided into two parts: $$π_i$$ and $$(i, t)$$, where $$π_i$$ is a shortest subpath from s to i. This decomposition is called the **optimality principle**.

```c
sort the nodes of G topologically
L_1 = 0;
for j=2,..,n do
	L_j = min{L_i + c_ij | (i, j) in δ−(j) AND i < j};
	pred_j = v such that (v, j) = argmin{L_i + c_ij | (i, j) in δ−(j) AND i < j};
```

The algorithm is exact due to the optimality principle. The dynamic programming is a general technique in which an optimal solution, composed of a sequence of elementary decisions, determined by solving a set of recursive equations. It is applicable to any sequential decision problem for which the optimality property is satisfied.

### Project planning

A projects consists of a set of m activities with their duration $$d_i ≥ 0$$, $$i = 1,..,m$$. Some pair of activities are subject to a **precedence constraint**: $$A_i ∝ A_j$$ indicates that $$A_j$$ can start only after the end of $$A_i$$. The arcs must be positioned such that there exists a directed path where the arc associated to $$A_i$$ precedes the one of $$A_j$$. A project can be represented by a directed graph where each arc corresponds to an activity, and the arc length represents the duration of the activity.

The directed graph representing a project is acyclic (DAG) by definition.

Given a project schedule the activities so as to minimize the overall project duration, that is the time needed to complete all the activities. The minimum project duration equal to the length of a longest path from s to t.

#### Critical path method (CPM)

Determines:

* schedule that minimize the overall project duration.
* the slack of each activity, that is the amount of time by which its execution can be delayed without affecting the overall minimum project duration.

Initialization: construct the graph G representing the project and find a topological order of the nodes. Consider the nodes by increasing indices and, for each $$h ∈ N$$ find: the earliest time $$T_{min_h}$$ at which the event associated to node h can occur (minimum project duration). Consider the nodes by decreasing indices and, for each $$h ∈ N$$ find: the latest time $$T_{max_h}$$ at which the event associated to node h can occur without delaying the project beyond the project completion date beyond $$T_{min_h}$$. For each activity $$(i, j) ∈ A$$, find the **slack** $$σ_{ij} = T_{max_j} − T_{min_i} − d_{ij}$$.

Input: graph $$G = (N, A)$$, with $$G = (N, A)$$, $$m = |A|$$ and the duration $$d_{ij}$$ associated to each $$(i, j) ∈ A$$. Output: $$(T_{min_i}, T_{max_i})$$, for $$i = 1, …, n$$.

```c
Sort the node topologically
T_min_1 = 0;
for j = 2,...,n do
	T_min_j = max{T_min_i + d_ij : (i, j) in δ−(j)}
T_max_n = T_min_n;
//in reverse order, from t to s
for i = n-1,...,1 do
	T_max_i = min{T_max_j - d_ij : (i, j) in δ+(i)}
```

An activity (i, j) with zero slack is critical. A critical path is an s → t path only composed of **critical activities** (one such path always exists).

### Network Flows

Problems regarding the distribution of a given product from a set of sources to a set of users so as to optimize a given objective function.

A network is a directed and connected graph $$G = (N, A)$$ with a source and a sink $$s,t ∈ N$$ with $$s ≠ t$$, and a capacity $$k_{ij} ≥ 0$$, for each arc $$(i, j) ∈ A$$.

A **feasible flow** x from s to t is a vector $$x ∈ R^m$$ with a component $$x_{ij}$$ for each arc $$(i, j) ∈ A$$ satisfying the capacity constraints $$0 ≤ x_{ij} ≤ k_{ij}$$, $$∀(i, j) ∈ A$$, and the flow balance constraint at each intermediate node $$u∈ N$$ ($$u ≠ s, t$$). The value of flow x is $$ϕ = \sum_{(s,j)∈δ+(s)}x_{sj}$$.

Given a network and a feasible flow x, an arc is **saturated** if $$x_{ij} = k_{ij}$$, otherwise it is **empty** if $$x_{ij} = 0$$.

#### Maximum flow problem

Given a network $$G = (N, A)$$ with an integer capacity $$k_{ij}$$ for each arc $$(i, j) ∈ A$$, and nodes $$s, t ∈ N$$, determine a feasible flow $$x ∈ R^m$$ from s to t of maximum value.

### Ford-Fulkerson’s algorithm

Idea: start from a feasible flow x and try to **iteratively increase** its **value** ϕ by sending, at each iteration, an additional amount of product along a path from s to t with a strictly positive residual capacity.

If $$(i, j)$$ is not saturated it is possible to increase $$x_{ij}$$, otherwise If $$(i,j)$$ is not empty it is possible to decrease $$x_{ij}$$ while respecting $$0 ≤ x_{ij} ≤ k_{ij}$$.

A path P from s to t is an **augmenting path** w.r.t the current feasible flow x if $$x_{ij} < k_{ij}$$ for every forward arc and $$x_{ij}>0$$ for every backward arc.

Given a feasible flow x for $$G = (N, A)$$, we construct the **residual network** $$G’ = (N, A’)$$ associated to x, accounting for all possible flow variations w.r.t. x:

* if $$(i, j) ∈ A$$ is not empty, then $$(i, j) ∈ A’$$ with $$k’{ji} = x{ij} > 0$$.
* if $$(i, j) ∈ A$$ is not saturated, then $$(i, j) ∈ A’$$ with $$k’{ji} = k{ij} - x_{ij} > 0$$.

Input: graph $$G = (N, A)$$ with capacity $$k_{ij} > 0$$, for every $$(i, j) ∈ A$$, $$s,t ∈ N$$. \
Output: feasible flow x from s to t of maximum value ϕ.

```c
x = 0;
ϕ = 0;
optimum = false;
do
	build residual network G' associated to x
	P = path from s to t in G'
	if P is not defined then 
		//no more additional units
		optimum = true;
	else 
		ϕ = min{k_ij : (i, j) in P};
		ϕ = ϕ + δ;
		for (i, j) in P do
			if (i, j) is a foward arc then 
				x_ij = x_ij + δ;
			else 
				x_ij = x_ij - δ;
while (optimum != true)
```

The Ford-Fulkerson’s algorithm is exact. It is not greedy. The value of a feasible flow of maximum value is equal to the capacity of a cut of minimum.

For maximum flow problems more efficient algorithms (in polynomial time) exists based on augmenting paths, pre-flows and capacity scaling. Idea: start from a feasible flow x of value ϕ and send, at each iteration, an additional amount of product in the residual network along cycles of negative cost.

#### Indirect applications

Given an undirected bipartite graph $$G = (N, E)$$, a matching $$M ⊆ E$$ is a subset of non adjacent edges. Given a bipartite graph the problem of determine a matching with a maximum number of edges can be reduced to the problem of finding a feasible flow of maximum value.

<figure><img src="/assets/Untitled 2 (2).png" alt=""><figcaption></figcaption></figure>

### Hard graph optimization problems

#### Traveling salesman problem (TSP)

Given a directed $$G = (N, A)$$ with a cost $$c_{ij}∈ Z$$ for each arc $$(i, j) ∈ A$$, determine a circuit of minimum total cost visiting every node exactly once. A Hamiltonian circuit C of G is a circuit that visits every node exactly once. Denoting by H the set of all Hamiltonian circuits of G, H contains a finite but exponential number of elements $$|H| ≤ (n-1)!$$.

#### NP-completeness theory

NP-hard computational problems can’t be solved by polynomial time algorithm (know to date). NP stands for non-deterministic polynomial.

## Linear Programming

A linear programming (LP) problem is an optimization problem min $$f(x)$$ s.t. $$x ∈ X ⊆ R^n$$, where the **objective function** $$f : X → R$$ is linear. $$x^* ∈ R^n$$ is an **optimal solution** if $$f(x^*) ≤ f(x), ∀x ∈ X$$.

![Untitled](/assets/Untitled.png)

Assumptions of LP models:

* **Linearity** of the objective function and constraints.
* **Divisibility**, the variables can take rational values.
* **Parameters** are considered as constants which can be estimated with a sufficient degree of accuracy.

### Equivalent forms

<figure><img src="/assets/Untitled 1.png" alt="" width="508"><figcaption></figcaption></figure>

The **standard** **form** has only equality constraints ($$Ax = b$$)and all non negative variables ($$x > 0$$). Simple **transformation rules** allow to pass from one form to the other:

<figure><img src="/assets/Untitled 2.png" alt="" width="545"><figcaption></figcaption></figure>

### Geometry of LP

A level curve of value z of a function $$f$$ is the set of points in $$R^n$$ where $$f$$ is constant and takes value z.

$$H = \{x ∈ R^n : a^tx = b \}$$ is a **hyperplane**. $$H^- = \{x ∈ R^n : a^tx< b\}$$ is an **affine half-space** (half-plane in $$R^2$$). Each inequality constraint of an LP problem defines an half-space in the variable space.

The feasible region X of any LP is a polyhedron P (intersection of finite number of half-planes). P can be empty or unbounded. A subset $$S ⊆ R^n$$ is convex if for each pair $$y_1, y_2 ∈ S$$, $$S$$ contains the whole segment connecting $$y_1$$ and $$y_2$$.

The segment defined by all the **convex combinations** of $$y_1$$ and $$y_2$$ is $$[y_1, y_2] = \{ x ∈ R^n : x = αy_1 + (1−α)y_2∧α ∈ [0, 1]\}$$. A polyhedron P is a convex set of $$R^n$$. Any half-space is convex, and the intersection of finite number of convex sets is also a convex set.

A vertex of P is a point of P which cannot be expresses as a convex combination of two other distinct points of P (either one is P itself).

A non-empty polyhedron $$P = \{x ∈ R^n : Ax = b, x ≥ 0 \}$$ (in standard form) or $$P = \{x ∈ R^n : Ax ≥ b, x ≥ 0 \}$$ (in canonical form) has infinite number of vertices.

Given a polyhedron P, a vector $$d ∈ R^n$$ with $$d ≠ 0$$ is an unbounded feasible direction of P if, for every point $$x_0 ∈ P$$, the “ray” $$\{x ∈ R^n: x = x_0 + λd, λ ≥ 0\}$$ is contained in P.

Every **point of a polyhedron** P can be expressed as a convex combination of its vertices $$x_1, …, x_k$$ plus (if needed) an unbounded feasible direction $$d$$ of P: $$x = α_1x_1 +...+ α_kx_k + d$$, where $$αi ≥ 0$$ satisfy $$α_1 + · · · + α_k = 1$$.

A **polytope** is a bounded polyhedron, that is when $$d = 0$$.

#### Fundamental theorem of LP

Consider a LP $$min\{c^Tx : x ∈ P \}$$ where $$P ⊆ R^n$$ is a non-empty polyhedron. Then either there exists (at least) one optimal vertex or the value of the objective function is unbounded below on P.

Geometrically: An interior point $$x ∈ P$$ cannot be an optimal solution → there always exists an improving direction. In an optimal vertex all feasible directions (for sufficiently small step) are “worsening” directions.

Types of linear programs:

<figure><img src="/assets/Untitled 4.png" alt="" width="563"><figcaption></figcaption></figure>

### Basic feasible solutions

Due to the LP fundamental theorem, to solve any LP it suffices to consider the vertices of the polyhedron P of the feasible solutions.

A vertex corresponds to the **intersections** of the **hyperplanes** associated to n inequalities. For any polyhedron $$P = \{ x ∈ R^n : Ax = b, x>0\}$$:

* the **facets** (edges in $$R^2$$) are obtained by setting one variable to 0.
* the vertices are obtained by setting $$n-m$$ variables to 0 (assuming $$A ∈ R^{m \times n}$$).

Algebraic characterization of the vertices: Consider any $$P = \{x ∈ R^n : Ax = b, x ≥ 0 \}$$ in standard form. Assuming that $$A ∈ R^{m×n}$$ is such that $$m ≤ n$$ of rank m (A is of full rank, equivalent to NO redundant constraints).

* if $$m=n$$, there is a unique solution of $$Ax = b$$ that is $$x = A^{-1}b$$.
* if $$m < n$$, there are ∞ solutions. The system has $$n − m$$ degrees of freedom (variables that can be fixed arbitrarily). By fixing them to 0, we get a vertex.

A basis of a matrix A is a subset of m columns of A that are linearly independent and form an m x m non singular matrix B.

<figure><img src="/assets/Untitled 5.png" alt="" width="551"><figcaption></figcaption></figure>

By construction $$(x_B^T, x_N^T)$$ together satisfy $$Ax = b$$.

A basic solution is obtained by setting $$x_n = 0$$ so that $$x_B = B^{-1}b$$. It is called basic feasible solution if $$x_B ≥ 0$$. The variables in $$x_B$$ are the basic variables, those in $$x_N$$ are the non basic variables. $$x ∈ R^n$$ is a basic feasible solution if and only if x is a vertex of the polyhedron P.

At most one basic feasible solution for each choice on the n-m non basic variables (to be set to 0) out of the n variables.

$$
\# basic\_feasible\_solutions \le \binom{n}{n-m} = \binom{n}{m}
$$

### Simplex method

Idea: examine a sequence of BFSs with **non-increasing objective function values** until an optimal solution is reached or the LP is found to be unbounded.

At each iteration, move from a BFS to a “neighboring” basic feasible solution. Generate a **path** (sequence of adjacent vertices) along the edges of the polyhedron of the feasible solutions until an optimal vertex is reached.

<figure><img src="/assets/Untitled 6.png" alt="" width="149"><figcaption></figcaption></figure>

#### Optimality test (determine if a current vertex is optimal)

Given a LP $$min\{ c^Tx : Ax = b, x≥0\}$$ and a feasible basis B of A, then $$Ax = B$$ can be rewritten as $$Bx_B + Nx_N = B \Rightarrow x_B = B^{-1}(b-Nx_N)$$ with $$B^{-1}b \ge 0$$.

<figure><img src="/assets/Untitled 7.png" alt="" width="563"><figcaption></figcaption></figure>

The vector of reduced costs w.r.t. the basis B (for BFSs) is $$c^T = c^T - c^T_B B^{-1}A = [\overbrace{c^T_B - c^T_B b^{-1}B}^{0^T}, \overbrace{c^T_N - c^T_BB^{-1}N}^{c^T_N}]$$

* The reduced costs are defined also for basic variables, but $$c_B = 0$$.
* $$c_j$$ represents the change in objective function value if non basic $$x_j$$ would be increased from 0 to 1 while keeping all other non basic variables to 0.
* The solution value changes by $$∆z = c_j∆x_j$$.

if $$c_N ≥ 0$$ then the basic feasible solution $$(x^T_B, x^T_N)$$, where $$x_B = B^{-1}b ≥ 0$$ and $$x_N = 0$$ of cost $$c^T_BB^{-1}b$$, is a global optimum.

This condition is sufficient but in general not necessary.

#### Move to an adjacent vertex (bfs)

Goal: improve the objective function value but preserve feasibility. \
Idea: when moving from the current vertex to an adjacent vertex, we substitute one column of B with one column of N.

Given a basis B, the system $$Ax = B \Leftrightarrow \sum^n_{j = 1}{a_{ij}x_j = b_i}$$ for $$i = 1,…,m$$ can be expressed in **canonical form** $$x_B + B^{-1}Nx_N = B^{-1}b \Leftrightarrow Bx_B + Nx_N = b$$ which emphasizes the basic feasible solution $$(x_B, x_N) = (B^{-1}b, 0)$$. In the canonical form the basic variables are expressed in terms of non basic variables.

**Change of basis (for min LP):** Let B be a feasible basis and $$x_s$$ (in $$x_N$$) a non basic variable with negative reduced cost $$c_s < 0$$.

* Increase $$x_s$$ as much as possibile while keeping the other non basic variables equal to 0. $$x_s$$ enters the basis.
* The basic variable $$x_r$$ (in $$x_B$$) that imposes the tightest upper bound $$θ^*$$ on the increase of $$x_s$$, leaves the basis.
* If $$θ^*>0$$, the new bfs has better objective function value. The new basis differs w.r.t. the previous one by a single column (represent adjacent vertices).

**Pivoting operation:**

* Given $$Ax = b$$ select a coefficient $$a_{rs} ≠ 0$$ (**pivot**).
* Divide the r-th row by $$a_{rs}$$.
* For each row i with $$i ≠ r$$ and $$a_{is} ≠ 0$$, subtract the resulting r-th row multiplied by $$a_{is}$$.

1. Which non basic variable enters the basis? Any one with negative reduced cost. One that gives the maximum $$\Delta z$$ w.r.t. $$z = c^T_B B^{-1}b$$. Bland’s rule.
2. Which basic variables leaves the basis? The index i with smallest $$b_i/a_{is} = θ^*$$, that imposes the tightest upper bound on increase of $$x_s$$, among those with $$a_{is} ≥0$$ (otherwise there is no limit).

**Unboundness**: if exist a non basic variable with negative reduced cost with $$a_{ij} ≤ 0, ∀i$$ (no element can play the role of a pivot) the minimization problem is unbounded.

#### “Tableau” representation

<figure><img src="/assets/Untitled 8.png" alt="" width="485"><figcaption></figcaption></figure>

<figure><img src="/assets/Untitled 9.png" alt="" width="563"><figcaption></figcaption></figure>

{% code overflow="wrap" %}
```c
//Simplex algorithm
Let B[1],..,B[m] be the column indices of the initial feasible basis B-
Construct the initial tableau A' = {a[i,j] : 0 <= i <= m, 0 <= j <= n} in canonical form w.r.t. B.
unbounded = false;
optimal = false;
while optimal == false and unbounded == false do
	//if all reduced cost of the non basic variables are non-negative the optimal is found
	if a[0, j] >= 0 forall j = 1,..,n then optimal = true;
	else 
		select a non basic variable x_s with a[0, s] < 0
		//if there is no element that can play the role of a pivot the problem is unbounded
		if a[i, s] <= 0 forall i = 1,..,m then unbounded = true;
		else 
				r = argmin(a[i, 0]/a[i, s] with 1 <= i <= m)
				pivot(r, s); //update the tableau by performig the pivot operation
				B[r] = s; //update the new basis

//The final tableau show the basis of the optimal solution

```
{% endcode %}

#### Degenerate BFSs and convergence

A basic feasible solution x is degenerate if it contains at least one basic variable $$x_j = 0$$.

In presence of degenerate BFSs, a basis change may not decrease the objective function value.

**Bland’s rule**: among all candidate variables to enter/exit the basis $$(x_s / x_r)$$ always select the one with smallest index. The Simplex algorithm with Bland’s rule terminates after ≤ $$\binom{n}{m}$$ iterations.

### Two-phase simplex method

Phase I: Determine an initial basic feasible solution.

<figure><img src="/assets/Untitled 10.png" alt="" width="563"><figcaption></figcaption></figure>

* Express the basic variables $$y_i$$ w.r.t. the non basic ones $$x_i$$.
* Write the objective function w.r.t the non basic variables $$x_i$$ and draw the initial tableau. Apply the simplex algorithm till the reduced costs for the non-basic variables are all non-negative.

If the LP auxiliary problem admits an optimal solution of value o then the original problems admits a BFS, and viceversa.

Phase II: Then starting from the obtained bases in phase I, apply the simplex algorithm on the original problem (P).

### Linear programming duality

To any minimization LP it is possible to associate a closely related maximization LP based on the same parameters, and viceversa. The original problem is called **primal** (**P**) the resulting one is called **dual** (**D**).

<figure><img src="/assets/Untitled 11.png" alt="" width="266"><figcaption></figcaption></figure>

General strategy: linearly combine the constraints with non negative multiplicative factors (i-th the one multiplied by $$y_i ≥ 0$$).

<figure><img src="/assets/Untitled 12.png" alt="" width="437"><figcaption></figcaption></figure>

#### Weak duality theorem

Given the previous (P) and (D) with $$X = \{x ∈ R^n : Ax≥ b, x≥0\} \ne 0$$ and $$Y = \{ y∈ R^m : A^Ty ≤ c, y≥0 \} ≠0$$. For every feasible solution $$x ∈ X$$ of (P) and $$y ∈ Y$$ of (D) we have $$b^Ty ≤ c^Tx$$.

As a consequence, if x is a feasible solution of (P) and y is a feasible solution of (D) and $$b^Ty = c^Tx$$, then x is optimal for (P) and y is optimal for (D).

#### Strong duality theorem

If $$X = \{x ∈ R^n : Ax≥ b, x≥0\} \ne 0$$ and $$min\{c^tx : x∈X\}$$ is finite, there exist $$x^* ∈ X$$ and $$y^*∈Y$$ (optimal solutions) such that $$c^Tx^* = b^Ty^*$$.

<figure><img src="/assets/Untitled 13.png" alt="" width="542"><figcaption></figcaption></figure>

#### Optimality conditions

Given the previous (P) and (D). Two feasible solutions $$x^* ∈ X$$ and $$y^* ∈ Y$$, with $$X =\{x ∈ R^n : Ax≥ b, x≥0\}$$ and $$Y = \{ y ∈ R^m : y^TA ≤ c^T, y≥ 0\}$$ are optimal if and only if $${y^*}^Tb= c^Tx^*$$

If $$x_j$$ and $$y_i$$ are unknown, this is a single equation in n+m unknowns. Since $${y^*}^T b≤ {y^*}^TAx^* ≤ {c^*}^Tx$$_, then we have_ $${y^*}^T = {y^*}^TAx^*$$ and $${y^*}^T Ax^* = {c^*}^Tx^*$$. These are m+n equations in n+m unknowns. Necessary and sufficient optimality conditions.

### Sensitivity analysis in LP

Evaluate the “**sensitivity**” of an optimal solution w.r.t. variations in the model parameters.

## Integer Linear Programming

An integer linear programming (ILP) is an optimization problem of the form min $$c^Tx$$ such that $$Ax ≥ b, x ≥ 0$$ with $$x ∈ Z^n$$ (integer values). We assume that the parameters A,b are integer (without loss of generality).

Given an ILP, the equal LP is the **linear** (continuous) **relaxation** of the ILP. For any ILP with max, we have $$z_{ILP} ≤ z_{LP}$$, that means that $$z_{LP}$$ is an u**pper bound** on the optimal value of ILP. Similarly for min. If an optimal solution of LP is integer, then it is also an optimal solution for ILP.

#### Knapsack problem

![binary version is NP-hard](</assets/Untitled (1).png>)

#### Assignment problem

<figure><img src="/assets/Untitled 1 (1).png" alt="" width="563"><figcaption></figcaption></figure>

#### Transportation problem

<figure><img src="/assets/Untitled 2 (1).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/Untitled 3 (1).png" alt="" width="563"><figcaption></figcaption></figure>

The assignment problem is a particular case of the transportation one. property of both: optimal solution of the linear relaxation ≡ optimal solution of the ILP.

If in a transportation problem $$p_i$$, $$d_{ij}$$, $$q_{ij}$$ are integer, all the BFSs (vertices) of its linear relaxation are integer.

#### Scheduling problem

<figure><img src="/assets/Untitled 4 (1).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/Untitled 5 (1).png" alt="" width="563"><figcaption></figcaption></figure>

ILP formulation can be extended to the case where each job j must be processed on a subset of the m machines according to a different order.

Most ILP problem are NP-hard, so there not exists efficient algorithms to solve them. Some methods:

* **implicit enumeration**: exact methods (global optimum). They explore all feasible solutions. “Branch-and-bound” or dynamic programming.
* **cutting planes**: exact methods (global optimum).
* **heuristic algorithms**: approximate methods (local optimum).

### Branch-and-bound method

Idea: reduce the solution of a difficult problem to that of a sequence of simpler sub-problems by (**recursive**) **partition** of the feasible region X. \
Two main components: **branching** and **bounding**. \
It is applicable to both discrete and continuous optimization problem.

<figure><img src="/assets/Untitled 6 (1).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/Untitled 7 (1).png" alt="" width="563"><figcaption></figcaption></figure>

Particular case for ILP min $$\{c^Tx : Ax = b, x>0\text{ integer}\}$$

* **Branching**: partition X into sub-regions. Let $$x’$$ be an optimal solution for the linear relaxation of the ILP, and $$z_{LP} = c^Tx’$$ the corresponding optimal value. if $$x’$$ is integer then it is also optimal for ILP. Otherwise $$∃x’_h$$ fractional that divide the original problem into:
  * $$ILP_1$$ : min $$\{c^Tx : Ax = b, x_h ≤ [x’_h], x ≥ 0 \text{ integer} \}$$
  * $$ILP_2$$ : min $$\{c^Tx : Ax = b, x_h ≥ [x’_h] + 1, x ≥ 0 \text{ integer} \}$$
* **Bounding**: determine a lower bound (if min ILP) on the optimal value $$z_i$$ of a sub-problem of ILP by solving its linear relaxation.

Branch-and-bound is an exact method, so it guarantees an optimal solution.

The branch tree may not contain all possibile nodes ( $$2^d$$ leaves). A node has no child (is “fathomed”) if:

* initial constraints + those on the arcs from the root are infeasible.
* optimal solution of the linear relaxation is integer.
* the value $$c^Tx’{LP}$$ _of the optimal solution_ $$x’{LP}$$ of the linear relaxation is worse than that of the best feasible solution of ILP found so far.

This **bounding criterion** allows to “discard” a large number of nodes (sub-problems).

#### Choice of the node to examine

* Deeper nodes first (depth-first strategy): simple recursive procedure, easy to re-optimize but it may be costly in case of the wrong choice.
* First more promising nodes with the best relaxation value (best-bound first strategy): generates smaller number of nodes but sub-problems are less constrained. It takes longer to find a first feasible solution and improve it.

#### Choice of the (fractional) variable for branching

* It may be not the best choice to select the variable $$x_{h}$$ whose fractional value is closer to 0.5, in order to obtain sub-problems that are more stringent and balanced.
* Strong branching: try to branch on some candidate variables (fractional basic ones), evaluate the corresponding objective function values and branch on the variable that yields the best improvement in objective function.

#### Efficient solution of the linear relaxations

* No need to solve the linear relaxation of the ILP from scratch (with the two-phase simplex algorithm).
* An optimal solution of the linear relaxation with a single additional constraint can be found via a single iteration of the simplex method applied to the dual to the optimal solution of the previous linear relaxation.

#### Remarks on branch-and-bound method

It is also applicable to mixed ILPs: when branching just consider the fractional variables that must be integer. Finding a good initial BFS with a heuristic may improve the method’s efficiency by providing a better lower bound $$z’$$ on $$z_{ILP}$$ (for the max).

### Cutting plane methods and gomory fractional cuts

Given a general ILP problem min $$\{c^Tx : Ax≥0, x≥0 \text{ integer} \}$$ with a feasible region $$X = \{x ∈ Z^n : Ax ≥ b, x>0 \}$$assuming that $$a_{ij}$$, $$c_j$$ and $$b_i$$ are integer. There are ∞ equivalent formulations of the feasible region but the optimal solutions of the linear relaxations can differ substantially.

The ideal formulation is that describing the convex hull of X $$conv(X)$$, that is the smallest convex subset containing X.

For any feasible region X of an ILP (bounded or unbounded), there exists an ideal formulation (a description of $$conv(X)$$ involving a finite number of linear constraints) but the number of constraints can be very large (exponential) w.r.t. the size of the original formulation.

A cutting plane is an inequality $$a^Tx ≤ b$$ that is not satisfied by $$x^*_{LP}$$ but is satisfied by all the feasible solutions of the ILP.

Idea: given an initial formulation, iteratively add cutting planes as long as the linear relaxation does not provide an optimal integer solution.

#### Gomory fractional cuts

Let $$x^*_{LP}$$ _be an optimal solution for the linear relaxation of the current formulation min_ $$\{ c^Tx : Ax = b, x≥ 0\}$$ _and_ $$x^{B[r]}$$ _be a fractional basic variable. The corresponding row of the optimal tableau is:_ $$X_{B[r]} + \sum_{j∈N}{a_{rj}x_j} = b’_r$$.

The Gomory cut w.r.t. the fractional basic variable $$x_{B[r]}$$ is: $$\sum_{j∈N}{a_{rj}- \lfloor a_{rj} \rfloor x_j} \ge (b'_r - \lfloor b'_r \rfloor)$$

* It is violated by the optimal fractional solution $$x^*_{LP}$$ of the linear relaxation: since ( $$b’_r - \lfloor b’_r \rfloor) > 0$$ and $$x_j = 0$$, $$∀j$$ s.t. $$x_j$$ non basic.
* It is satisfied by all integer feasible solutions:

<figure><img src="/assets/Untitled 9 (2).png" alt="" width="541"><figcaption></figcaption></figure>

The “**integer**” **form** $$x_{B[r]} + \sum_{j∈N}{a_{rj}x_j} \le \lfloor b'r \rfloor$$ _and the “**fractional**” **form**_ $$\sum{j∈N}{a_{rj}- \lfloor a_{rj} \rfloor x_j} \ge (b'_r - \lfloor b'_r \rfloor)$$ of the cutting plane are equivalent.

```c
Solve the linear relaxation min{c^Tx : Ax = b, x >= 0}
Let x' be an optimal BFS of the linear relaxation
While x' has fractional components do:
	select a basic variable with fractional value
	Generate the corresponding Gomory cut
	Add constraint to the optimal tableau of the linear relaxation
	Perform one iteration of the dual simplex algorithm
```

If the ILP has finite optimal solution, the cutting plane method finds one after adding a finite number of gomory cuts.

#### Branch-and-cut

Idea: combine the branch-and-bound with the cutting plane method in order to overcome the disadvantages of the two. For each sub-problem of the B\&B, several cutting planes are generated to improve the bound and try to find an optimal integer solution. Whenever the cutting planes become less effective, cut generation is stopped and a branching operation is performed.

Advantages: the cuts tend to strengthen the formulation of the sub-problems. The long series of cuts without sensible improvement are interrupted by branching operations.

