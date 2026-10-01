log = Log.open_topic ("clearvoice-lock")

local lock_enabled = Settings.get_boolean ("clearvoice.lock-base-mic-audio")
local base_mic_node = Settings.get_string ("clearvoice.base-mic-node") or ""
local base_mic_gain = Settings.get_float ("clearvoice.base-mic-gain")
local output_lock_enabled = Settings.get_boolean ("clearvoice.lock-output-volume")
local mixer = nil

local audio_sources = ObjectManager {
  Interest {
    type = "node",
    Constraint { "media.class", "=", "Audio/Source", type = "pw-global" },
  }
}

local function source_exists (node_name)
  return audio_sources:lookup (
    Interest { type = "node", Constraint { "node.name", "=", node_name } }) ~= nil
end

local function lock_active ()
  return lock_enabled and source_exists ("clearvoice_source")
end

local permission_manager = PermissionManager ()
permission_manager:set_default_permissions (Perm.ALL)
permission_manager:add_interest_match (
  function (_, _, object)
    local node_name = object.properties["node.name"]
    local hide = lock_active ()
      and (node_name == base_mic_node or node_name == "clearvoice_beamformed")
    return hide and Perm.NONE or Perm.ALL
  end,
  Interest {
    type = "node",
    Constraint { "media.class", "=", "Audio/Source" },
  }
)

local function refresh_permissions ()
  permission_manager:update_permissions ()
end

local function enforce_gain ()
  if mixer == nil then
    return
  end

  for node in audio_sources:iterate () do
    local name = node.properties["node.name"]
    local target = nil
    if lock_active () and name == base_mic_node then
      target = base_mic_gain
    elseif output_lock_enabled and name == "clearvoice_source" then
      target = 1.0
    end
    if target ~= nil then
      local id = node["bound-id"]
      local current = mixer:call ("get-volume", id)
      if current ~= nil and math.abs (current.volume - target) > 0.005 then
        mixer:call ("set-volume", id, { volume = target })
      end
    end
  end
end

Settings.subscribe ("clearvoice.*", function ()
  lock_enabled = Settings.get_boolean ("clearvoice.lock-base-mic-audio")
  base_mic_node = Settings.get_string ("clearvoice.base-mic-node") or ""
  base_mic_gain = Settings.get_float ("clearvoice.base-mic-gain")
  output_lock_enabled = Settings.get_boolean ("clearvoice.lock-output-volume")
  refresh_permissions ()
  enforce_gain ()
end)

audio_sources:connect ("object-added", function ()
  refresh_permissions ()
  enforce_gain ()
end)
audio_sources:connect ("object-removed", refresh_permissions)

SimpleEventHook {
  name = "client/find-clearvoice-access",
  before = { "client/find-default-access", "client/apply-access" },
  after = {
    "client/find-config-access",
    "client/find-flatpak-access",
    "client/find-snap-access",
    "client/find-portal-access",
  },
  interests = {
    EventInterest {
      Constraint { "event.type", "=", "select-access" },
    },
  },
  execute = function (event)
    if event:get_data ("permission-manager") ~= nil
        or event:get_data ("default-permissions") ~= nil then
      return
    end

    local client = event:get_subject ()
    local properties = client.properties
    local access = properties["pipewire.access"]
      or properties["pipewire.client.access"]
    if properties["clearvoice.client"] == "true"
        or properties["pipewire.sec.socket"] == "pipewire-0-manager"
        or properties["pipewire.sec.flatpak"] ~= nil
        or properties["pipewire.snap.id"] ~= nil
        or (access ~= nil and access ~= "unrestricted") then
      return
    end

    event:set_data ("permission-manager", permission_manager)
  end
}:register ()

audio_sources:activate ()

mixer = Plugin.find ("mixer-api")
if mixer ~= nil then
  mixer["scale"] = "cubic"
  mixer:connect ("changed", function () enforce_gain () end)
  enforce_gain ()
else
  log:error ("mixer API is unavailable")
end
