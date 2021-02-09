local eps = 0
local maxiter = 40000

local function vector(len, v)
	local x = {}
	for i = 1, len do x[i] = v end
	return x
end

local function norm(v)
	local n = 0
	for i, _v in ipairs(v) do
		n = n + (_v * _v)
	end
	return math.sqrt(n)
end

local function gauss_seidel(m, b)
	local n = #m
	local x = vector(n, 0)
	local q, p, sum
	local t = 0

	repeat
		t = t + 1
		q = norm(x)
		for i = 1, n do
			sum = 0
			for j = 1, n do
				if (i ~= j) then
					sum = sum + m[i][j] * x[j]
				end
			end
			x[i] = (1 / m[i][i]) * (b[i] - sum)
		end
		p = norm(x)
	until (math.abs(p - q) < eps) or (t >= maxiter)
	return x
end

return gauss_seidel
