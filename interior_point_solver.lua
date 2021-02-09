local linalg = require(script:WaitForChild("linalg"))
local gs_solver = require(script:WaitForChild("gauss_seidel"))

function norm(m)
	local n = 0
	for i,v in ipairs(m) do
		n = n + v^2
	end
	return math.sqrt(n)
end

local function mat_to_table(matrix)
	local new_matrix = {}
	for x=1,matrix.Shape[1] do
		new_matrix[x] = {}
		for y=1,matrix.Shape[2] do
			new_matrix[x][y] = matrix[x][y]
		end
	end
	return new_matrix
end

local function mat_recip(matrix)
	local new_matrix = {}
	for x=1,matrix.Shape[1] do
		new_matrix[x] = {}
		for y=1,matrix.Shape[2] do
			new_matrix[x][y] = 1 / matrix[x][y]
		end
	end
	return linalg.matrix.new(new_matrix)
end

local ipm_init = function(A, B, C, init_value)
	local self = {}

	init_value = init_value or 0

	self.mat_a = A
	self.vec_b = B
	self.vec_c = C

	self.init_vec = function(size, how_many)
		local vec = {}
		for i=1, how_many do
			table.insert(vec, linalg.matrix.new({
				table.create(size, init_value)
			}))
		end

		return unpack(vec)
	end

	self.get_step_length = function(x, s, delta_x, delta_s, theta)
		local alpha_x = 1
		local alpha_s = 1
		local alpha_k = 1

		for i=1, self.mat_a.Shape[2] do
			if delta_x[i] < 0 then
				alpha_x = math.min(alpha_x, -x[i] / delta_x[i])
			end
			if delta_s[i] < 0 then
				alpha_s = math.min(alpha_s, -s[i] / delta_s[i])
			end
		end

		alpha_x = math.min(1.0, theta * alpha_x)
		alpha_s = math.min(1.0, theta * alpha_s)
		alpha_k = math.min(1.0, theta *math. min(alpha_s, alpha_x))

		return alpha_x, alpha_s, alpha_k
	end

	self.solve = function(theta, gamma, epsilon)		
		local n = self.mat_a[1].Length
		local x, s, e = self.init_vec(n, 3)
		local y = linalg.matrix.new({
			table.create(self.mat_a.Shape[1], 0)
		})
		local k = 0

		local x_iterations = {}


		while (x * s.T)[1][1] > epsilon do
			print("------------------")
			if k % 100 == 1 then wait() end
			table.insert(x_iterations, x)

			local r_primal = self.vec_b - (self.mat_a * x.T).T
			local r_dual = self.vec_c - (self.mat_a.T * y.T).T - s

			local mu_k = (x * s.T) / n
			local s_recip_ = mat_recip(s)

			local s_recip = linalg.matrix.diagonal(s_recip_[1]._values)
			local x_ = linalg.matrix.diagonal(x[1]._values)

			local m_ = ((self.mat_a * s_recip) * x_) * self.mat_a.T

			local r_1 = self.mat_a*s_recip
			local r_2 = (x_ * r_dual.T).T - (gamma*mu_k*e)
			local r = (self.vec_b.T + (r_1 * r_2.T)).T
			
			local m_table = mat_to_table(m_)
			local r_table = r[1]._values

			local delta_y_sol_table = gs_solver(m_table, r_table)

			local delta_y = linalg.matrix.new({delta_y_sol_table})
			local delta_s = r_dual - (self.mat_a.T * delta_y.T).T
			local delta_x = (s_recip * ((gamma*mu_k*e) - (x_ * delta_s.T).T).T).T - x

			local alpha_x, alpha_s, alpha_k = self.get_step_length(x[1]._values, s[1]._values, delta_x[1]._values, delta_s[1]._values, theta)

			x = x + alpha_x * delta_x
			y = y + alpha_k * delta_y
			s = s + alpha_s * delta_s

			local objective_value = (self.vec_c * x.T)[1][1]
			
			print(k, "Objective Value: ", objective_value)	
			
			k = k + 1
		end
		
		return x
	end

	return self
end

local A = linalg.matrix.new({
	{1, 0, 1, 0, 0, 0, 0},
	{2, 2, 0, 1, 0, 0, 0},
	{4, 1, 0, 0, 1, 0, 0},
	{4, 2, 0, 0, 0, 1, 0},
	{1, 2.2, 0, 0, 0, 0, 1}
})

local B = linalg.matrix.new({
	{2.3, 10, 10, 12, 10}
})

local C = linalg.matrix.new({
	{-1, -2, 0, 0, 0, 0, 0}
})
ipm = ipm_init(A, B, C, 1.0)
local x_result = ipm.solve(0.99, 1e-3, 1e-8)
print("----------------")
print("---- RESULT -----")
print(x_result[1][1], x_result[1][2])
print("----------------")
