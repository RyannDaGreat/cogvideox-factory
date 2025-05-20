TODO: Make the noise dataloader generalize as follows:

#Somewhere on the top of the datasets file we IMOPRT this from auxiliary_datatypes import auxiliary_datatypes or something like that...
auxiliary_datatypes = {
	"noise" : {
		["noise.txt", "noises.txt"],
		torch.load,
	},
	"canny" : {
		["noise.txt", "noises.txt"],
		lambda image_path: canny(load_image(image_path)),
	},
	"mask" : {
		["mask.txt", "masks.txt"],
		torch.load,
	},
}

And an argument passed to the dataset like this:
--auxiliary_data "noises canny" # Etc space separated - include only the ones you need. It will raise an error IF AND ONLY IF required_auxiliary_data is specified, but can't be found.

This is meant to generalizze the noise dataloading we have now to future datatypes.


NOTE:
The auxiliary datatypes loading funcs might be specified in the config...OR...more elegantly, they can just have a different name...
    For example: 
    "on_the_fly_warped_noise" : {
        ["noise.txt", "noises.txt"]
    }
    <---- RIGHT NOW WE CAN'T ACTUALLY DO THAT!!
