def compare_with_dcgan(stylegan_path, dcgan_path, output_dir='comparison'):
    """
    Generate samples from both models for visual comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    stylegan = StyleGANGenerator().to(device)
    stylegan.load_state_dict(torch.load(stylegan_path, map_location=device)['generator_state_dict'])
    stylegan.eval()

    dcgan = DCGANGenerator().to(device)
    dcgan.load_state_dict(torch.load(dcgan_path, map_location=device)['generator_state_dict'])
    dcgan.eval()

    with torch.no_grad():
        
        z = torch.randn(16, 512, device=device)

        
        stylegan_samples = stylegan(z)
        save_image(stylegan_samples, f'{output_dir}/stylegan_samples.png', nrow=4, normalize=True)

        
        z_dcgan = z.unsqueeze(2).unsqueeze(3) if len(dcgan(z[:1]).shape) == 4 else z
        dcgan_samples = dcgan(z_dcgan)
        save_image(dcgan_samples, f'{output_dir}/dcgan_samples.png', nrow=4, normalize=True)

    print(f"Comparison samples saved to {output_dir}")