
function initialize(box)

	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	delay = box:get_setting(2)
	iteration = box:get_setting(3)
	increment = box:get_setting(4) == "true"


end

function process(box)
	local t = 0
	local stimulation = OVTK_StimulationId_Label_01

	-- manages Timeout
	for i=1,iteration do 
		box:send_stimulation(1, stimulation, t, 0)
		t = t + delay
		if increment then stimulation = stimulation + 0x00000001 end
	end
end
