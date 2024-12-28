
function initialize(box)

	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	-- each stimulation sent that gets rendered by Display Cue Image box 
	-- should probably have a little period of time before the next one or the box wont be happy
	pre_baseline_duration = 1
	baseline_duration = 10
	post_baseline_duration = 1
	cross_duration = 1
	post_cross_duration = 1
	display_cue_duration = 3
	post_cue_duration = 3
	rest_duration = 5
	post_end_duration = 1

	sequence = {
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_01,
	}
	
end

function process(box)

	local t = 0

	-- Delays before the trial sequence starts
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStart, t, 0)
	t = t + pre_baseline_duration

	box:send_stimulation(1, OVTK_StimulationId_BaselineStart, t, 0)
	t = t + baseline_duration

	box:send_stimulation(1, OVTK_StimulationId_BaselineStop, t, 0)
	t = t + post_baseline_duration

	-- creates each trial
	for i = 1, #sequence do

		-- first display a cross on screen
		box:send_stimulation(1, OVTK_GDF_Start_Of_Trial, t, 0)
		box:send_stimulation(1, OVTK_GDF_Cross_On_Screen, t, 0)
		box:send_stimulation(1, OVTK_StimulationId_Beep, t, 0)	
		t = t + cross_duration

		-- Clear cross. 
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStop, t, 0)
		t = t + post_cross_duration
		
		-- display cue
		box:send_stimulation(1, sequence[i], t, 0)
		t = t + display_cue_duration

		-- clear cue. 
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStop, t, 0)
		t = t + post_cue_duration

		-- rest period
		box:send_stimulation(1, OVTK_StimulationId_RestStart, t, 0)
		t = t + rest_duration	
		
		-- end of rest and trial
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStop, t, 0)
		box:send_stimulation(1, OVTK_StimulationId_RestStop, t, 0)
		box:send_stimulation(1, OVTK_GDF_End_Of_Trial, t, 0)
		t = t + post_end_duration	
	end

	-- send end for completeness	
	box:send_stimulation(1, OVTK_GDF_End_Of_Session, t, 0)
	t = t + 5

	-- used to cause the acquisition scenario to stop and denote final end of file
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, t, 0)
	
end
