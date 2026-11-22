--[[
    Off-line harness for the AudioStreamWriter path in rsc_decoder.lua.

    Stubs the engine just enough to run the decoder without Roblox: a ring buffer that drains in
    real time, Float32 sample capture, and a virtual clock so 6 seconds of audio can be played in
    a fraction of a second. Writes the captured samples to /tmp/lua_stream.f32 as
    float32 little-endian, channel 0 first then channel 1.

        luajit /tmp/drive_stream.lua audio.rsc2 rsc_decoder.lua
]]

local ffi = require("ffi")
local SR  = 48000

local payloadPath = arg[1]
local decoderPath = arg[2]

-- ── virtual clock and the audio engine it feeds ─────────────────────────────
local T          = 0.0
local consumed   = 0          -- frames the "engine" has drained
local written    = 0          -- frames the decoder has written
local RING       = 8192       -- frames of ring buffer, so latency is about 170 ms

local sink, sinkCap = {}, 0
local function growSink(need)
    if need <= sinkCap then return end
    local cap = math.max(need, 1 << 20)
    for ch = 1, 2 do
        local old = sink[ch]
        sink[ch] = ffi.new("float[?]", cap)
        if old and sinkCap > 0 then
            ffi.copy(sink[ch], old, sinkCap * 4)
        end
    end
    sinkCap = cap
end
sink[1] = ffi.new("float[?]", 1 << 20)
sink[2] = ffi.new("float[?]", 1 << 20)
sinkCap = 1 << 20

-- ── engine stubs ────────────────────────────────────────────────────────────
-- LuaJIT ships `bit`, not `bit32`, and its rshift is arithmetic rather than logical.
local bit = require("bit")
bit32 = {
    bor    = bit.bor,
    band   = bit.band,
    lshift = function(v, n) return bit.lshift(v, n) end,
    rshift = function(v, n)
        local r = bit.rshift(v, n)
        if v < 0 then r = r + bit.lshift(1, 32 - n) end
        return r
    end,
}

buffer = {}
table.create = function(n) return {} end
function buffer.create(n)
    return {f = ffi.new("float[?]", math.max(math.floor(n/4), 1)), bytes = n}
end
function buffer.writef32(b, off, v) b.f[math.floor(off/4)] = v end
function buffer.readf32(b, off) return b.f[math.floor(off/4)] end

local ASW = {}
ASW.__index = ASW
function ASW:GetOutputPins() return {"Output"} end
function ASW:Space()
    local free = RING - (written - consumed)
    if free < 0 then return 0 end
    return free
end
function ASW:Write(buffers)
    local n = math.floor(buffers[1].bytes/4)
    growSink(written + n)
    for ch = 1, #buffers do
        ffi.copy(sink[ch] + written, buffers[ch].f, n * 4)
    end
    written = written + n
    return n
end

Instance = {}
local Instances = {}
function Instances.Destroy() end
function Instances.Play() end
function Instances.IsA() return false end

function Instance.new(cls, parent)
    if cls == "AudioStreamWriter" then
        return setmetatable({Layout = 0, WindowSize = 1, SampleRate = SR,
                             SampleFormat = 1}, ASW)
    end
    return setmetatable({Name = cls, Parent = parent}, {__index = Instances})
end

workspace = {FindFirstChild = function() return nil end}

Enum = {
    AudioChannelLayout = {Mono = 0, Stereo = 1},
    AudioWindowSize    = {Small = 0, Medium = 1, Large = 2},
    AudioSampleFormat  = {Int16 = 0, Float32 = 1},
}

game = {GetService = function() return {WaitForChild = function() return {} end} end}
task = {wait = function() end}
warn = function(msg) print("[warn] " .. tostring(msg)) end

script = {
    IsA = function() return false end,
    Parent = {FindFirstChild = function() return nil end,
              GetFullName = function() return "harness" end},
}

local clock = {}
function clock.get() return T end
os.clock = clock.get

-- ── load the decoder and the payload ───────────────────────────────────────
local f = assert(io.open(payloadPath, "rb"))
local payload = f:read("a")
f:close()
print(string.format("payload %s bytes", #payload))

local RSC2 = loadfile(decoderPath)()

local dec = assert(RSC2.new(payload))
print("engine: " .. dec.engine.name)

dec:play()

-- 120 ticks a second of virtual time, which is how a Heartbeat-driven client behaves
local hopTick = os.getenv("TICK_HOPS")
local dt, totalSec = (hopTick and (1024 / SR)) or (1 / 120),
              tonumber(os.getenv("SECONDS")) or 6.0
local dumpPath = os.getenv("SLOT_DUMP")
local dump, lastHop = nil, -1
if dumpPath then
    dump = assert(io.open(dumpPath, "w"))
end
local lastApplied = {}
for ch = 1, dec.channels do lastApplied[ch] = -2 end
local tick = 0
local function snapEveryTick()
    if dump == nil then return end
    tick = tick + 1
    for ch = 1, dec.channels do
        local reg = dec.engine.reg
        if reg then
            dump:write(string.format("F %d %d %d\n", tick, ch, dec.engine.applied[ch]))
            for bin, s in pairs(reg[ch]) do
                dump:write(string.format("T %d %d %d %.10f %.10f %.10f\n", tick, ch, bin, s.amp, s.target, s.ph))
            end
        end
    end
end
local function snap()
    if dump == nil then return end
    for ch = 1, dec.channels do
        local applied = dec.engine.applied and dec.engine.applied[ch] or -1
        if applied >= 0 and applied ~= lastApplied[ch] then
            lastApplied[ch] = applied
            local reg = dec.engine.reg[ch]
            for bin, s in pairs(reg) do
                dump:write(string.format("%d %d %d %.10f %.10f %.10f\n",
                                         applied, ch, bin, s.amp, s.target, s.ph))
            end
        end
    end
end
local steps = math.floor(totalSec / dt)
local peakPartials = 0
for i = 1, steps do
    T = i * dt
    consumed = math.floor(T * SR)
    dec:tick(dt)
    local p = dec.engine:partialCount()
    if p > peakPartials then peakPartials = p end
    snapEveryTick()
end
T = totalSec
dec:stop()

print(string.format("played %.2f s, wrote %d frames (%.2f s of audio), peak partials %d, clips %d",
                    totalSec, written, written / SR, peakPartials, dec.engine.clips))
if written ~= math.floor(totalSec * SR) then
    print(string.format("NOTE: expected %d frames", math.floor(totalSec * SR)))
end

local out = assert(io.open(os.getenv("OUT") or "/tmp/lua_stream.f32", "wb"))
for ch = 1, dec.channels do
    out:write(ffi.string(sink[ch], written * 4))
end
out:close()
print("wrote /tmp/lua_stream.f32")


print('done')
