% OV_MATLAB toolkit
% OV_define.m
% -------------------------------
% Author : Laurent Bonnet (INRIA)
% Date   : 28/07/2011

% This file is meant to be executed prior to any user defined script.
% It declare all stim IDs as  variable for use  by the user.
% MEANT TO BE CALLED BY OPENVIBE
% 
% Stimulation codes 
% Originally from openvibe-toolkit/ovtk_defines.h


function OV_define()

% STREAM TYPES:
    global OV_TypeId_EBMLStream;
    OV_TypeId_EBMLStream = uint64(hex2dec('434F65872EFD2B7E'));
    global OV_TypeId_ChannelLocalisation;
    OV_TypeId_ChannelLocalisation = uint64(hex2dec('013DF452A3A8879A'));
    global OV_TypeId_ExperimentationInformation;                                % deprecated
    OV_TypeId_ExperimentationInformation = uint64(hex2dec('403488E7565D70B6')); % deprecated
    global OV_TypeId_ExperimentInfo;
    OV_TypeId_ExperimentInfo = uint64(hex2dec('403488E7565D70B6'));
    global OV_TypeId_Stimulations;
    OV_TypeId_Stimulations = uint64(hex2dec('6F752DD0082A321E'));
    global OV_TypeId_StreamedMatrix;
    OV_TypeId_StreamedMatrix = uint64(hex2dec('544A003E6DCBA5F6'));
    global OV_TypeId_FeatureVector;
    OV_TypeId_FeatureVector = uint64(hex2dec('17341935152FF448'));
    global OV_TypeId_Signal;
    OV_TypeId_Signal = uint64(hex2dec('5BA36127195FEAE1'));
    global OV_TypeId_Spectrum;
    OV_TypeId_Spectrum = uint64(hex2dec('1F261C0A593BF6BD'));

% SETTING TYPES:
	global OV_TypeId_Boolean;
    OV_TypeId_Boolean = uint64(hex2dec('2CDB2F0B12F231EA'));
	global OV_TypeId_Integer;
    OV_TypeId_Integer = uint64(hex2dec('007DEEF92F3E95C6'));
	global OV_TypeId_Float;
    OV_TypeId_Float = uint64(hex2dec('512A166F5C3EF83F'));
	global OV_TypeId_String;
    OV_TypeId_String = uint64(hex2dec('79A9EDEB245D83FC'));
	global OV_TypeId_Filename;
    OV_TypeId_Filename = uint64(hex2dec('330306DD74A95F98'));
	global OV_TypeId_Script;
    OV_TypeId_Script = uint64(hex2dec('B0D0DB4549CBC34A'));
	global OV_TypeId_Stimulation;
    OV_TypeId_Stimulation = uint64(hex2dec('2C132D6E44AB0D97'));
	global OV_TypeId_LogLevel;
    OV_TypeId_LogLevel = uint64(hex2dec('A88B36670871638C'));
	global OV_TypeId_Color;
    OV_TypeId_Color = uint64(hex2dec('7F45A2A97DB12219'));
	global OV_TypeId_ColorGradient;
    OV_TypeId_ColorGradient = uint64(hex2dec('3D3C7C7FEF0E7129'));
	
	
% STIM CODES:
    global OVTK_StimulationId_LabelStart;
     OVTK_StimulationId_LabelStart                         = uint64(hex2dec('00008100'));
    global OVTK_StimulationId_LabelEnd;
     OVTK_StimulationId_LabelEnd                           = uint64(hex2dec('000081ff'));
     
    global OVTK_StimulationId_NumberStart;
     OVTK_StimulationId_NumberStart                         = uint64(hex2dec('00000000'));
    global OVTK_StimulationId_NumberEnd;
     OVTK_StimulationId_NumberEnd                           = uint64(hex2dec('000000ff'));
end
