
g_sent = false
g_numTrials = 10

function initialize(box)
	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")
	-- make sure we get equal number of both trials
 	g_numTrials = 2 * box:get_setting(2)
	g_sent = false;
end

function uninitialize(box)
end

function process(box)

	while box:keep_processing() and g_sent == false do

		box:send_stimulation(1, OVTK_StimulationId_ExperimentStart, 0, 0)	
	
		current_time = 0

		for i = 1 , g_numTrials do

			if i % 2 == 0 then
				box:send_stimulation(1, OVTK_StimulationId_NonTarget,             current_time+0, 0)
			else
				box:send_stimulation(1, OVTK_StimulationId_Target,                current_time+0, 0)
			end

			current_time = current_time + 1

		end
		
		box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, current_time, 0)	
        box:send_stimulation(1, OVTK_StimulationId_EndOfFile, current_time+2, 0)

		g_sent = true
		
		box:sleep()
		
	end
	
end

