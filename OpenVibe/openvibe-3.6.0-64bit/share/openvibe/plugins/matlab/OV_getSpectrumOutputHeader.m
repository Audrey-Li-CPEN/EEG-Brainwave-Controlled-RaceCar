% openvibe toolbox for Matlab
% OV_getSpectrumOutputHeader.m
% -------------------------------
% Author : Laurent Bonnet (INRIA)
% Date   : 28/07/2011

% Set the header information on a signal output.
% MEANT TO BE USED IN USER SCRIPT

function [errno, nb_channels, channel_names, nb_abscissas, abscissas_names, abscissas_linear, sampling_rate] = OV_getSpectrumOutputHeader(box_in, output_index)
   
    if(numel(box_in.outputs{output_index}.header) == 0)
        nb_channels = 0;
        channel_names = 0;
		nb_abscissas = 0;
		abscissas_names = 0;
		abscissas_linear = 0;
		sampling_rate = 0;
        errno = 1;
    else
        nb_channels = box_in.outputs{output_index}.header.nb_channels;
        channel_names = box_in.outputs{output_index}.header.channel_names;
		
		nb_abscissas = box_in.outputs{output_index}.header.nb_abscissas;
		abscissas_names = box_in.outputs{output_index}.header.abscissas_names;
		abscissas_linear = box_in.outputs{output_index}.header.abscissas;
		%reshape(box_in.outputs{output_index}.header.bands,1,numel(box_in.outputs{output_index}.header.bands));
		sampling_rate = box_in.outputs{output_index}.header.sampling_rate;
        errno = 0;
    end    
end
