--[[
    RSC_Decoder.lua  —  LocalScript (StarterGui or StarterPlayer.StarterPlayerScripts)

    Plays RSC2 audio in real time. Two engines, picked automatically at startup:

      STREAM  AudioStreamWriter, engine 747 and newer. Luau writes Float32 samples into a ring
              buffer that the audio engine drains at 48 kHz. This is the good one: the decoder
              owns the waveform, so every partial is a real oscillator with the exact bin
              frequency and phase that accumulates without ever being reset, and there is no
              per-Sound parameter traffic to make clicks.

      BANK    Looped 1 kHz Sounds with PlaybackSpeed, for engines older than 747 and for the
              cases where AudioStreamWriter is not available. Same frame decoder, same ramping
              idea, phase lives on the asset's own timeline so it can only be nudged.

    WHY AN AMPLITUDE RAMP IS ENOUGH
    A Hann window shifted by half its length sums to exactly 1.0, so two frames that agree about
    a partial, windowed and added together, are that partial running continuously. Overlap-add
    only differs from a continuous oscillator where amplitude or frequency changes, and a ramp is
    what overlap-add is doing there anyway. So: continuous phase plus a half-hop amplitude ramp
    gets the same result as the Python decoder's overlap-add, without the window multiply.

    WHAT THE DOCUMENTATION SAYS ABOUT AudioStreamWriter
      - SampleRate is 48000 always, SampleFormat is Float32 always, both read-only.
      - One Output pin, no inputs. It must be wired to an AudioDeviceOutput or nothing is heard.
      - Layout sets how many buffers Write() wants: Stereo takes two, one per channel.
      - Space() reports frames that fit without dropping. Write() returns frames actually taken.
      - Roblox's docs state it is usable in Studio only and not in Team Create sessions, which is
        the whole reason the bank engine is still here.

    Accepts RSC2 v1 and v2. v1 frames butt together because that file carries no overlap, so its
    frame joins still click; re-encode to v2 to get rid of that.
]]

local Players           = game:GetService("Players")
local RunService        = game:GetService("RunService")

local tcreate   = table.create
local floor     = math.floor
local clamp     = math.clamp or function(v, lo, hi)   -- Roblox ships math.clamp, plain Lua does not
	if v < lo then return lo elseif v > hi then return hi end
	return v
end
local TAU       = 2 * math.pi
local SIN_N     = 4096                          -- wavetable points, power of two
local AMP_EPS   = 1e-5                          -- quieter than this is not worth an oscillator
local CHUNK     = 1024                          -- frames per Write, one hop, about 21 ms
local SPIN_MAX  = 16                            -- chunks in one tick before we give up and wait
local LEAD_MAX  = 0.20                          -- underrun longer than this, skip forward
local SINE_ASSET_ID = "rbxassetid://113823829598029"
local SINE_ASSET_HZ = 1000                      -- the looped asset is a 1 kHz tone

-- ════════════════════════════════════════════════════════════════════════════
-- BINARY READING
-- ════════════════════════════════════════════════════════════════════════════

local function readU16LE(data, off)
	return bit32.bor(string.byte(data, off), bit32.lshift(string.byte(data, off + 1), 8))
end

local function readU32LE(data, off)
	return bit32.bor(
		string.byte(data, off),
		bit32.lshift(string.byte(data, off + 1), 8),
		bit32.lshift(string.byte(data, off + 2), 16),
		bit32.lshift(string.byte(data, off + 3), 24)
	)
end

local function readF32LE(data, off)
	local bits = bit32.bor(
		string.byte(data, off),
		bit32.lshift(string.byte(data, off + 1), 8),
		bit32.lshift(string.byte(data, off + 2), 16),
		bit32.lshift(string.byte(data, off + 3), 24)
	)
	local sign = bit32.rshift(bits, 31)
	local exp  = bit32.band(bit32.rshift(bits, 23), 0xFF)
	local mant = bit32.band(bits, 0x7FFFFF)
	if exp == 0 then
		return 0
	elseif exp == 255 then
		return math.huge
	end
	local value = (1 + mant / 0x800000) * math.pow(2, exp - 127)
	return sign == 1 and -value or value
end

-- ════════════════════════════════════════════════════════════════════════════
-- WAVETABLE
-- ════════════════════════════════════════════════════════════════════════════
-- math.sin every sample is a native call every sample. A lookup with linear interpolation is a
-- handful of Lua operations instead, and that is what makes 48 kHz affordable in Luau. Linear
-- interpolation on 4096 points sits around -75 dB, well under anything this codec keeps.

local SIN = tcreate(SIN_N + 2)
for i = 1, SIN_N do
	SIN[i] = math.sin((i - 1) * TAU / SIN_N)
end
SIN[SIN_N + 1] = SIN[1]                         -- wrap the end so idx + 1 always exists
SIN[SIN_N + 2] = SIN[2]

-- ════════════════════════════════════════════════════════════════════════════
-- FRAME ITERATOR
-- ════════════════════════════════════════════════════════════════════════════
-- Both engines read frames the same way. This walks a frame's records in place instead of
-- building a table per partial per frame, because on the stream path this runs 47 times a second
-- per channel and allocation is the cost that shows up first.

local function eachPartial(dec, frameIdx, ch, fn)
	local off = dec.frameBase + ((ch - 1) * dec.nFrames + frameIdx) * dec.frameStride
	local peak  = readF32LE(dec.data, off)
	local nPart = readU16LE(dec.data, off + 4)

	local p = off + 6
	for _ = 1, nPart do
		local bin   = readU16LE(dec.data, p)
		local ampU  = readU16LE(dec.data, p + 2)
		local phase = (readU16LE(dec.data, p + 4) / 65535) * TAU - math.pi
		p = p + 6

		-- rfft bins hold half a tone, DC and Nyquist a whole one. Missing that costs 6 dB.
		local scale = (bin > 0 and bin < dec.nyquist) and 2 or 1
		local amp   = (ampU / 65535) * peak * scale / dec.windowSum
		if amp > AMP_EPS and bin > 0 and bin < dec.nyquist then
			fn(bin, bin * dec.binHz, amp, phase)
		end
	end
end

-- ════════════════════════════════════════════════════════════════════════════
-- ENGINE 1: AUDIOSTREAMWRITER
-- ════════════════════════════════════════════════════════════════════════════

local StreamEngine = {}
StreamEngine.__index = StreamEngine
StreamEngine.name = "stream (AudioStreamWriter)"

local function deviceOutput()
	if workspace == nil then
		return nil
	end
	local dev = workspace:FindFirstChild("RSC2_AudioDeviceOutput")
	if not dev then
		dev = Instance.new("AudioDeviceOutput")
		dev.Name = "RSC2_AudioDeviceOutput"
		dev.Parent = workspace
	end
	return dev
end

-- nil means not available, and the caller falls back to the bank.
function StreamEngine.tryNew(dec)
	if dec.sampleRate ~= 48000 then
		warn(string.format("[RSC2/stream] file is %d Hz but AudioStreamWriter runs at 48000, "
			.. "using the bank", dec.sampleRate))
		return nil
	end

	local ok, writer = pcall(Instance.new, "AudioStreamWriter")
	if not ok or writer == nil then
		return nil
	end

	local self = setmetatable({}, StreamEngine)
	self.dec         = dec
	self.channels    = math.max(1, math.min(dec.channels, 2))
	self.hop         = dec.hopSize
	self.fft         = dec.fftSize
	self.totalFrames = dec.nSamples               -- header stores samples per channel
	self.pos         = 0                          -- frames generated, so also the position
	self.running     = false
	self.dead        = false
	self.clips       = 0
	self.shortWrites = 0
	self.sumBuf      = table.create(dec.hopSize)  -- scratch for the partial-outer render

	local setup = pcall(function()
		if self.channels == 2 then
			writer.Layout = Enum.AudioChannelLayout.Stereo
		else
			writer.Layout = Enum.AudioChannelLayout.Mono
		end
		writer.WindowSize = Enum.AudioWindowSize.Medium
	end)
	if not setup then
		warn("[RSC2/stream] could not set Layout/WindowSize")
	end

	self.writer = writer
	-- Graph nodes must be in the DataModel tree to resolve pins.
	writer.Parent = workspace

	-- Without a sink the stream exists but nobody hears it, so this is a hard requirement.
	local sink = deviceOutput()
	if sink == nil then
		warn("[RSC2/stream] no workspace, cannot wire an output")
		pcall(function() writer:Destroy() end)
		return nil
	end
	local wire = Instance.new("Wire")
	wire.Name = "RSC2_Wire"
	wire.SourceInstance = writer
	pcall(function()
		local pins = writer:GetOutputPins()
		if pins and pins[1] then
			wire.SourceName = pins[1]
		end
	end)
	wire.TargetInstance = sink
	wire.TargetName = "Input"
	wire.Parent = writer
	self.wire = wire
	if not wire.Connected then
		warn("[RSC2/stream] wire not connected; graph may be broken")
	end

	-- A bin is a frequency, so a bin owns its slot for the life of the partial. No identity churn,
	-- no re-sorting by amplitude. reg[bin] is the slot, list[] is what the mix loop walks.
	self.reg   = {}
	self.list  = {}
	self.applied = {}
	for ch = 1, self.channels do
		self.reg[ch]     = {}
		self.list[ch]    = {}
		self.applied[ch] = -1
	end

	-- τ = half a hop, the same time constant the bank engine uses so both sound alike.
	local tau = (dec.hopSize / 2) / 48000
	self.rampK = 1 - math.exp(-(1 / 48000) / tau)

	print(string.format("[RSC2/stream] AudioStreamWriter | 48000 Hz | %d ch | tau %.2f ms",
		self.channels, 1000 * (dec.hopSize / 2) / 48000))
	return self
end

function StreamEngine:destroy()
	self.dead = true
	pcall(function() self.writer:Destroy() end)
	pcall(function() self.wire:Destroy() end)
end

function StreamEngine:muteAll()
	for ch = 1, self.channels do
		self.applied[ch] = -1
		for _, slot in pairs(self.reg[ch]) do
			slot.target = 0
		end
	end
end

function StreamEngine:start()
	self.startWall = os.clock()
	self.running   = true
end

function StreamEngine:stop()
	self.running = false
	self:muteAll()
end

function StreamEngine:seek(seconds)
	seconds = clamp(seconds, 0, self.dec.duration)
	self.pos       = math.floor(seconds * 48000)
	self.startWall = os.clock() - seconds
	self:muteAll()
end

-- Pulls one frame into the slot register: new bins arrive at gain 0 and ramp in, missing bins
-- leave at a ramp instead of being cut.
function StreamEngine:applyFrame(ch, frameIdx)
	local reg  = self.reg[ch]
	local seen = self.seenScratch

	if seen == nil then
		seen = {}
		self.seenScratch = seen
	end
	for k in pairs(seen) do
		seen[k] = nil
	end

	local self_, fft = self, self.fft
	eachPartial(self.dec, frameIdx, ch, function(bin, freq, amp, phase)
		local slot = reg[bin]
		if slot == nil then
			slot = {
				bin    = bin,
				inc    = bin / fft,           -- turns per sample, so freq is exactly bin * binHz
				amp    = 0,
				target = amp,
				ph     = (phase / TAU) % 1,   -- seeded once; after that phase only accumulates
			}
			reg[bin] = slot
		else
			slot.target = amp
		end
		seen[slot] = true
	end)

	local list, n = self.list[ch], 0
	for bin, slot in pairs(reg) do
		if seen[slot] == nil then
			slot.target = 0
		end
		if slot.target > 0 or slot.amp > AMP_EPS then
			n = n + 1
			list[n] = slot
		else
			reg[bin] = nil
		end
	end
	for i = #list, n + 1, -1 do
		list[i] = nil
	end
end

-- Renders `frames` samples per channel straight into Float32 buffers.
function StreamEngine:render(frames)
	local dec, hop = self.dec, self.hop
	local rampK, SIN_TAB = self.rampK, SIN
	local pos, clips = self.pos, self.clips

	local buffers = tcreate(self.channels)
	for ch = 1, self.channels do
		buffers[ch] = buffer.create(frames * 4)
	end

	local done = 0
	while done < frames do
		local frameIdx = floor(pos / hop)
		if frameIdx >= dec.nFrames then
			frameIdx = dec.nFrames - 1
		end
		for ch = 1, self.channels do
			if self.applied[ch] ~= frameIdx then
				self.applied[ch] = frameIdx
				self:applyFrame(ch, frameIdx)
			end
		end

		local seg = math.min(frames - done, hop - (pos % hop))

		for ch = 1, self.channels do
			local list  = self.list[ch]
			local count = #list
			local buf   = buffers[ch]
			local base  = done * 4
			local sumBuf = self.sumBuf

			for k = 1, seg do
				sumBuf[k] = 0
			end

			-- One 3-term recurrence roll per partial instead of floor + two table reads
			-- per sample: s(m+1) = 2 cos(w) s(m) - s(m-1). Seeded from the wavetable once
			-- per hop, so drift re-seeds before it can accumulate.
			for j = 1, count do
				local s   = list[j]
				local inc = s.inc
				local ph  = s.ph
				local b1  = math.sin((ph + inc) * TAU)
				local b0  = math.sin(ph * TAU)
				local w   = 2 * math.cos(TAU * inc)
				local amp, target = s.amp, s.target
				for k = 1, seg do
					amp = amp + (target - amp) * rampK
					sumBuf[k] = sumBuf[k] + amp * b1
					local nb = w * b1 - b0
					b0, b1 = b1, nb
				end
				s.ph  = (ph + inc * seg) % 1
				s.amp = amp
			end

			for k = 1, seg do
				local sum = sumBuf[k]
				if sum > 1 then
					sum, clips = 1, clips + 1
				elseif sum < -1 then
					sum, clips = -1, clips + 1
				end
				buffer.writef32(buf, base + (k - 1) * 4, sum)
			end
		end

		pos  = pos + seg
		done = done + seg
	end

	self.pos   = pos
	self.clips = clips
	return buffers
end

function StreamEngine:tick(dt)
	if self.dead or not self.running then
		return
	end
	if self.pos >= self.totalFrames then
		self:stop()
		return
	end

	local elapsed = os.clock() - self.startWall
	local behind  = elapsed * 48000 - self.pos
	if behind > LEAD_MAX * 48000 then
		-- We could not feed the ring buffer in time. Skipping forward is better than an
		-- ever-longer make-up queue, which turns into a burst and then a click.
		warn(string.format("[RSC2/stream] underrun of %.2f s, jumping to real time",
			behind / 48000))
		self.pos = math.floor(elapsed * 48000)
		self:muteAll()
	end

	local need = floor(elapsed * 48000) - self.pos
	if need <= 0 then
		return
	end

	-- One Write takes a hop at most, and a 60 fps tick is only 800 frames at 48 kHz, so writing
	-- once per tick is not enough. Keep filling until caught up, the buffer is full, or we have
	-- spun enough to know something is wrong.
	local spin = SPIN_MAX
	while need > 0 and spin > 0 do
		local space
		local ok, res = pcall(function() return self.writer:Space() end)
		if not ok then
			self:fail(res)
			return
		end
		space = res or 0

		local n = math.min(need, CHUNK, space)
		if n < 1 then
			break                               -- buffer full, that is the engine pacing us
		end

		local buffers  = self:render(n)
		local ok2, got = pcall(function() return self.writer:Write(buffers) end)
		if not ok2 then
			self:fail(got)
			return
		end
		got = got or 0

		-- render() moved self.pos by n, so hand back whatever the engine did not take.
		self.pos = self.pos - (n - got)
		if got < n then
			self.shortWrites = self.shortWrites + 1
			if self.shortWrites > 32 then
				self:fail("Write() keeps coming up short, the ring buffer never drains")
			end
			break
		end
		self.shortWrites = 0
		need = need - got
		spin = spin - 1
	end
end

function StreamEngine:fail(err)
	if self.dead then
		return
	end
	self.dead = true
	warn("[RSC2/stream] giving up, the bank engine takes over: " .. tostring(err))
	pcall(function() self.writer:Destroy() end)
	pcall(function() self.wire:Destroy() end)
end

function StreamEngine:position()
	return self.pos / 48000
end

function StreamEngine:partialCount()
	local n = 0
	for ch = 1, self.channels do
		n = n + #self.list[ch]
	end
	return n
end

-- ════════════════════════════════════════════════════════════════════════════
-- ENGINE 2: SINE BANK
-- ════════════════════════════════════════════════════════════════════════════

local SinePool = {}
SinePool.__index = SinePool

function SinePool.new(size, hopSec)
	local self = setmetatable({}, SinePool)
	self.size    = size
	self.hopSec  = hopSec
	self.tau     = hopSec / 2
	self.sounds  = tcreate(size)
	self.bins    = tcreate(size)
	self.binSlot = {}
	self.target  = tcreate(size)

	local folder = Instance.new("Folder")
	folder.Name   = "RSC2_SineBank"
	folder.Parent = workspace
	self.folder = folder

	for i = 1, size do
		local s = Instance.new("Sound")
		s.Name          = "sine" .. i
		s.SoundId       = SINE_ASSET_ID
		s.Looped        = true
		s.Volume        = 0
		s.PlaybackSpeed = 1
		s.Parent        = folder
		s:Play()
		self.sounds[i] = s
		self.bins[i]   = 0
		self.target[i] = 0
	end
	return self
end

-- Exponential ramp toward the frame's gain. Writing Volume directly on every frame is what made
-- the joins click, so nothing here is allowed to step.
function SinePool:ramp(dt)
	local k = 1 - math.exp(-dt / self.tau)
	for i = 1, self.size do
		local target = self.target[i]
		local now    = self.sounds[i].Volume
		if target ~= now then
			self.sounds[i].Volume = now + (target - now) * k
		end
	end
end

function SinePool:applyFrame(partials)
	local seen = {}
	for _, p in ipairs(partials) do
		local slot
		for i = 1, self.size do
			if self.bins[i] == p.bin then
				slot = i
				break
			end
		end
		if slot == nil then
			for i = 1, self.size do
				if self.bins[i] == 0 then
					slot = i
					break
				end
			end
		end
		if slot == nil then
			break                                   -- bank full, the rest of the frame waits
		end
		if self.bins[slot] ~= p.bin then
			self.bins[slot] = p.bin
			local s = self.sounds[slot]
			-- Retuning resamples the asset and nudges phase, so write the speed here and the
			-- phase only now, while this slot is still silent.
			s.PlaybackSpeed = p.freq / SINE_ASSET_HZ
			s.TimePosition  = ((p.phase / TAU) % 1) / SINE_ASSET_HZ
			self.target[slot] = 0
		end
		seen[slot] = true
		self.target[slot] = p.amp
	end
	for i = 1, self.size do
		if seen[i] == nil and self.bins[i] ~= 0 then
			self.target[i] = 0
			if self.sounds[i].Volume < AMP_EPS then
				self.bins[i] = 0                    -- faded to silence, slot is free again
			end
		end
	end
end

function SinePool:fadeToSilence()
	for i = 1, self.size do
		self.target[i] = 0
	end
end

function SinePool:activeCount()
	local n = 0
	for i = 1, self.size do
		if self.bins[i] ~= 0 then
			n = n + 1
		end
	end
	return n
end

local BankEngine = {}
BankEngine.__index = BankEngine
BankEngine.name = "bank (looped Sounds)"

function BankEngine.new(dec)
	local self = setmetatable({}, BankEngine)
	self.dec     = dec
	self.pool    = SinePool.new(math.max(math.min(dec.maxPartials, 192), 1), dec.hopSec)
	self.frame   = -1
	self.running = false
	return self
end

function BankEngine:start()
	self.startWall = os.clock()
	self.running   = true
	self.frame     = -1
end

function BankEngine:stop()
	self.running = false
	self.frame   = -1
	self.pool:fadeToSilence()
end

function BankEngine:seek(seconds)
	seconds = clamp(seconds, 0, self.dec.duration)
	self.startWall = os.clock() - seconds
	self.frame = -1
	if not self.running then
		self:render(math.floor(seconds / self.dec.hopSec))
	end
end

function BankEngine:render(frameIdx)
	if frameIdx == self.frame then
		return
	end
	self.frame = frameIdx
	if frameIdx < 0 or frameIdx >= self.dec.nFrames then
		self.pool:fadeToSilence()
		return
	end
	local partials = tcreate(self.dec.maxPartials)
	local n = 0
	eachPartial(self.dec, frameIdx, 1, function(bin, freq, amp, phase)
		n = n + 1
		partials[n] = {bin = bin, freq = freq, amp = amp, phase = phase}
	end)
	self.pool:applyFrame(partials)
end

function BankEngine:tick(dt)
	if self.running then
		local elapsed = os.clock() - self.startWall
		if elapsed >= self.dec.duration then
			self:stop()
		else
			self:render(math.floor(elapsed / self.dec.hopSec))
		end
	end
	self.pool:ramp(dt)
end

function BankEngine:position()
	if not self.running then
		return 0
	end
	return clamp(os.clock() - self.startWall, 0, self.dec.duration)
end

function BankEngine:partialCount()
	return self.pool:activeCount()
end

function BankEngine:destroy()
	pcall(function() self.pool.folder:Destroy() end)
end

-- ════════════════════════════════════════════════════════════════════════════
-- RSC2 DECODER
-- ════════════════════════════════════════════════════════════════════════════

local RSC2 = {}
RSC2.__index = RSC2

function RSC2.new(rscData)
	local self = setmetatable({}, RSC2)

	if string.sub(rscData, 1, 4) ~= "RSC2" then
		error("not an RSC2 file")
	end
	self.data = rscData

	self.version      = string.byte(rscData, 5)
	self.channels     = string.byte(rscData, 6)
	self.sampleRate   = readU32LE(rscData, 7)
	self.fftSize      = readU16LE(rscData, 11)
	self.hopSize      = readU16LE(rscData, 13)
	self.maxPartials  = readU16LE(rscData, 15)
	self.nFrames      = readU32LE(rscData, 17)
	self.nSamples     = readU32LE(rscData, 21)
	self.windowSum    = readF32LE(rscData, 25)

	if self.hopSize <= 0 or self.fftSize <= 0 or self.sampleRate <= 0 then
		error("implausible fft/hop/sample_rate in header")
	end
	if self.windowSum <= 0 then
		self.windowSum = self.fftSize
	end

	self.hopSec      = self.hopSize / self.sampleRate
	self.binHz       = self.sampleRate / self.fftSize
	self.nyquist     = math.floor(self.fftSize / 2)
	self.frameBase   = 4 + 24 + 1                 -- 4 magic + 24 header, 1-based
	self.frameStride = 6 + self.maxPartials * 6
	self.duration    = self.nFrames * self.hopSec

	print(string.format("[RSC2] v%d | %d Hz | %d ch | %d frames | fft=%d hop=%d | %.2f s",
		self.version, self.sampleRate, self.channels, self.nFrames,
		self.fftSize, self.hopSize, self.duration))
	if self.version < 2 then
		warn("[RSC2] predates the overlap-add fix: frame joins will still click. "
			.. "Re-encode this file to v2.")
	end

	local engine = StreamEngine.tryNew(self)
	if engine then
		self.engine = engine
	else
		warn("[RSC2] AudioStreamWriter unavailable, playing through the Sound bank")
		self.engine = BankEngine.new(self)
	end
	return self
end

-- Kept for anything that wants the frame as data, mostly the tests.
function RSC2:readFrame(frameIdx, ch)
	local partials = tcreate(self.maxPartials)
	local n = 0
	eachPartial(self, frameIdx, ch, function(bin, freq, amp, phase)
		n = n + 1
		partials[n] = {bin = bin, freq = freq, amp = amp, phase = phase}
	end)
	return partials
end

function RSC2:play()
	self.engine:start()
end

function RSC2:stop()
	self.engine:stop()
end

function RSC2:seek(seconds)
	self.engine:seek(seconds)
end

function RSC2:tick(dt)
	self.engine:tick(dt or 0)
end

function RSC2:currentSecond()
	return self.engine:position()
end

-- ════════════════════════════════════════════════════════════════════════════
-- GUI
-- ════════════════════════════════════════════════════════════════════════════

local function createGUI(decoder)
	local screenGui        = Instance.new("ScreenGui")
	screenGui.Name         = "RSC_DecoderGUI"
	screenGui.ResetOnSpawn = false
	screenGui.Parent       = Players.LocalPlayer:WaitForChild("PlayerGui")

	local frame            = Instance.new("Frame")
	frame.Size             = UDim2.new(0, 460, 0, 96)
	frame.Position         = UDim2.new(0, 16, 1, -112)
	frame.BackgroundColor3 = Color3.fromRGB(18, 18, 22)
	frame.BorderSizePixel  = 0
	frame.Parent           = screenGui
	Instance.new("UICorner", frame).CornerRadius = UDim.new(0, 8)

	local function button(text, x)
		local b = Instance.new("TextButton")
		b.Size             = UDim2.new(0, 90, 0, 30)
		b.Position         = UDim2.new(0, x, 0, 8)
		b.Text             = text
		b.AutoButtonColor  = true
		b.BackgroundColor3 = Color3.fromRGB(48, 52, 64)
		b.TextColor3       = Color3.fromRGB(235, 235, 240)
		b.Font             = Enum.Font.GothamMedium
		b.TextSize         = 14
		b.Parent           = frame
		Instance.new("UICorner", b).CornerRadius = UDim.new(0, 6)
		return b
	end

	local play = button("Play", 8)
	local stop = button("Stop", 102)
	local back = button("-5s",  196)
	local fwd  = button("+5s",  290)

	local info         = Instance.new("TextLabel")
	info.Size                  = UDim2.new(1, -16, 0, 44)
	info.Position              = UDim2.new(0, 8, 0, 46)
	info.BackgroundTransparency = 1
	info.TextColor3            = Color3.fromRGB(180, 200, 220)
	info.Font                  = Enum.Font.Gotham
	info.TextSize              = 13
	info.TextWrapped           = true
	info.TextXAlignment        = Enum.TextXAlignment.Left
	info.Text                  = ""
	info.Parent                = frame

	play.MouseButton1Click:Connect(function()
		if decoder.engine.running then
			decoder:stop()
		else
			decoder:play()
		end
	end)
	stop.MouseButton1Click:Connect(function() decoder:stop() end)
	back.MouseButton1Click:Connect(function() decoder:seek(decoder:currentSecond() - 5) end)
	fwd.MouseButton1Click:Connect(function() decoder:seek(decoder:currentSecond() + 5) end)

	return screenGui, info
end

-- ════════════════════════════════════════════════════════════════════════════
-- MAIN
-- ════════════════════════════════════════════════════════════════════════════

local function main()
	-- Receive the fetched .rsc2 payload from the server's RSC_Fetcher via the
	-- RSC_DataEvent RemoteEvent (no BinaryStringValue needed).
	local ReplicatedStorage = game:GetService("ReplicatedStorage")

	local event = nil
	local deadline = os.clock() + 10
	repeat
		event = ReplicatedStorage:FindFirstChild("RSC_DataEvent")
		if not event then task.wait(0.1) end
	until event or os.clock() > deadline
	if not event then
		warn("[RSC2] RSC_DataEvent RemoteEvent not found in ReplicatedStorage (is RSC_Fetcher running?)")
		return
	end

	local payload = nil
	local fetchError = nil
	local gotData = false
	local conn = event.OnClientEvent:Connect(function(ok, data)
		gotData = true
		if ok and type(data) == "string" and #data > 0 then
			payload = data
		else
			fetchError = tostring(data)
		end
	end)

	-- Tell the server we're listening, in case it fired the data before this
	-- script connected (the fetcher re-sends on request).
	event:FireServer()

	deadline = os.clock() + 15
	while not gotData and os.clock() < deadline do
		task.wait(0.1)
	end
	conn:Disconnect()

	if payload == nil then
		warn("[RSC2] no payload from RSC_Fetcher: " .. (fetchError or "timed out waiting for RSC_DataEvent"))
		return
	end

	local built, decoder = pcall(RSC2.new, payload)
	if not built then
		warn("[RSC2] decode failed: " .. tostring(decoder))
		return
	end

	local _, info = createGUI(decoder)
	RunService.Heartbeat:Connect(function(dt)
		decoder:tick(dt)
		local e = decoder.engine
		info.Text = string.format(
			"%.2f / %.2f s   %s\npartials %d   clipped samples %d",
			decoder:currentSecond(), decoder.duration, e.name,
			e:partialCount(), e.clips or 0)
	end)
end

-- Runs itself when it is a real LocalScript in the DataModel, so the same file can be loaded as
-- a module for tests.
if script and script:IsA("LocalScript") then
	main()
end

return RSC2