from torch import nn, optim
from util import *
from models import *
from train_util import denorm_obj, norm_layerfeat_func
import copy

def parse_dnn_def(dnn_path, device, log_layerfeat=False, norm_layerfeat=False, norm_layerfeat_option='', norm_path=''):
    dnn_def = parse_json(dnn_path)
    
    dnn_def_tensor = []
    for layer_def in dnn_def:
        if norm_layerfeat:
            layer_def = norm_layerfeat_func(layer_def, log_layerfeat, norm_layerfeat, norm_layerfeat_option, norm_path=norm_path)
        layer_feat_batch_tensor = torch.Tensor(layer_def).to(device)
        dnn_def_tensor.append(layer_feat_batch_tensor)

    num_predictors = len(dnn_def)
    return dnn_def_tensor, num_predictors


def gen_dnn_predictors(model, num_predictors):
    model.latency_predictors = []
    model.energy_predictors = []
    for i in range(num_predictors):
        new_model_latency = copy.deepcopy(model.predictor)
        model.latency_predictors.append(new_model_latency)
        
        new_model_energy = copy.deepcopy(model.predictor_energy)
        model.energy_predictors.append(new_model_energy)


def visualize_recon(model, test_data, train_data, epoch, args):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    zip_test_data = list(zip(*test_data[:10]))
    zip_train_data = list(zip(*train_data[:10]))
    zip_all_data = zip_test_data + zip_train_data
    for i, (g, y, energy) in enumerate(zip_all_data):
        g_recon = model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)
        plot_config(g, args.res_dir, name0, data_type=args.data_type)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_config(g_recon, args.res_dir, name1, data_type=args.data_type)


def eval_arch(arch_config_lst):
    sys.path.insert(1, '/scratch/qijing.huang/cosa/src/')
    # from bo import eval
    from bo import eval

    # Evaluate arch config, given as a list
    # Can throw an error
    if torch.is_tensor(arch_config_lst):
        arch_config_lst = [t.item() for t in arch_config_lst]

    dir_name = "eval_arch" + args.save_appendix
    try:
        os.makedirs(dir_name)
    except Exception as e:
        pass
    arch_config_path = pathlib.Path(dir_name).resolve() / "_".join([str(t) for t in arch_config_lst])

    # Runtime config
    base_arch_dir = f"{os.environ['COSA_DIR']}/configs/arch/simba_dse.yaml"
    arch_dir = pathlib.Path("dse_arch_motivation_5").resolve()
    output_dir = pathlib.Path("output_dir_motivation_5").resolve()
    dataset_path = output_dir / f'dataset.csv'
    use_cache = False

    if use_cache and arch_config_path.exists():
        with open(arch_config_path, "rb") as f:
            eval_result = pickle.load(f)
    else:
        print("Evaluating:", arch_config_lst)
        # Change to cosa-vae directory to run eval
        cwd = os.getcwd()
        os.chdir('/scratch/charleshong/cosa-vae/src/')

        eval_result = None
        try:
            # eval_result = eval(arch_config_lst, base_arch_dir, arch_dir, output_dir, arch_v3=False, unique_sum=True, workload_dir=None)
            eval_result = eval(arch_config_lst, base_arch_dir, arch_dir, output_dir, dataset_path)
        except Exception as e:
            print(e)
        
        # Return to original directory    
        os.chdir(cwd)

        # Store eval result, or None if eval failed
        with open(arch_config_path, "wb") as f:
            pickle.dump(eval_result, f)

    return eval_result

def get_percent_diff(y_batch, y_pred, metric):
    y_batch = np.array(y_batch)
    y_pred = np.array(y_pred)
    # diff = torch.div(torch.abs(torch.sub(y_batch, y_pred)), y_batch)
    diff = np.divide(np.absolute(y_batch-y_pred), y_batch)
    # diff_sum = torch.sum(diff, 0) 
    diff_sum = np.sum(diff, 0) 
    diff_per_entry = diff_sum / diff.shape[0]
    # diff_per_entry = diff_sum / diff.size()[0]
    print(f'{metric} diff_per_entry: {diff_per_entry}')
    return diff_sum

def test_model():
    model.eval()
    real_img_manual = torch.tensor([
                                    [32, 18, 112,33110, 1742, 121960]
                                   ],
                                   dtype=torch.float64).to(device)
    labels_manual = torch.tensor([4121961]).to(device)

    mu, logvar = model.encode(real_img_manual)
    z = model.reparameterize(mu, logvar)

    print("mu, logvar:")
    print(mu, logvar)

    print("z:")
    print(z)

    decoded = model.decode(z)
    print("decoded:")
    print(decoded)

    # Test decoding
    plot_range = 2

    # plot_pred = ""
    plot_pred = "edp"
    input_name = "pred_cycle"
    input_idx = 0 # Should be 0 for "cycle" or "energy"

    # L1 = [1, 1.1, 1.2, 1.3, 1.4]
    # L2 = [2.9, 3, 3.1, 3.2, 3.3]
    #L1 = np.linspace(-plot_range, plot_range, 40) # Latent dim 1
    L1 = np.linspace(2.5,3, 40) # Latent dim 1
    #L2 = np.linspace(-plot_range, plot_range, 40) # Latent dim 2
    L2 = np.linspace(0,0.5, 40) # Latent dim 2
    X, Y = np.meshgrid(L1, L2)  # Grid for contour3D purposes
    Z = np.zeros(X.shape) # Holds grid of decoded values
    z = [None] * (X.shape[0] * X.shape[1]) # Holds latent vectors to decode

    # [16, 1024, 1024, 64, 16384, 1024, 65536] -> [0.1280, -0.4072,  1.3486,  1.1342]
    for yidx in range(X.shape[0]):
        for xidx in range(X.shape[1]):
            zidx = yidx*X.shape[1] + xidx
            z[zidx] = [ X[yidx][xidx],  Y[yidx][xidx], 0, 0]
            # z = torch.tensor([[ X[yidx][xidx],  Y[yidx][xidx], 0, 0]]).double()
    
    if plot_pred == "cycle":
        decoded = model.predictor(torch.tensor(z).float()).to(device) * 2**28
    elif plot_pred == "energy":
        decoded = model.predictor_energy(torch.tensor(z).float().to(device)) * 2**38
    elif plot_pred == "edp":
        cycle = model.predictor(torch.tensor(z).float()) * 2**28
        energy = model.predictor_energy(torch.tensor(z).float()) * 2**38
        decoded = energy * cycle
    else:
        decoded = model.decode(torch.tensor(z).float().to(device))

    for yidx in range(X.shape[0]):
        for xidx in range(X.shape[1]):
            zidx = yidx*X.shape[1] + xidx
            arch_val = decoded[zidx][input_idx]
            Z[yidx][xidx] = arch_val

    print(Z[X.shape[0]-1][X.shape[1]-1])

    # print(Z[0][0])
    # exit(0)

    # arch_val = decoded[0][input_idx]
    # Z[yidx][xidx] = arch_val

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(np.array(L1), np.array(L2), np.array(Z), 60, cmap='binary')
    #ax.set_xlabel('l1')
    #ax.set_ylabel('l2')
    #ax.set_zlabel(input_name)

    # ax.view_init(25, 35)
    # plt.savefig('contour_l1_l2.png', bbox_inches='tight')

    ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), cmap='viridis', edgecolor='none')
    plt.savefig('contour_l1_l2_trisurf.png', bbox_inches='tight')
    # exit(0)

    #L3 = np.linspace(-plot_range, plot_range, 40)
    L3 = np.linspace(0.5,1, 40)
    #L4 = np.linspace(-plot_range, plot_range, 40)
    L4 = np.linspace(4,4.5, 40)
    X, Y = np.meshgrid(L3, L4)
    Z = np.zeros(X.shape)
    z = [None] * (X.shape[0] * X.shape[1]) # Holds latent vectors to decode

    # [16, 1024, 1024, 64, 16384, 1024, 65536] -> [0.1280, -0.4072,  1.3486,  1.1342]
    for yidx in range(X.shape[0]):
        for xidx in range(X.shape[1]):
            zidx = yidx*X.shape[1] + xidx
            #z[zidx] = [X[yidx][xidx], Y[yidx][xidx]] #[ 0, 0, X[yidx][xidx], Y[yidx][xidx]]
            z[zidx] = [0, 0, X[yidx][xidx], Y[yidx][xidx]]
            
    if plot_pred == "cycle":
        decoded = model.predictor(torch.tensor(z).float().to(device)) * 2**28
    elif plot_pred == "energy":
        decoded = model.predictor_energy(torch.tensor(z).float().to(device)) * 2**38
    elif plot_pred == "edp":
        cycle = model.predictor(torch.tensor(z).float()) * 2**28
        energy = model.predictor_energy(torch.tensor(z).float()) * 2**38
        decoded = energy * cycle
    else:
        decoded = model.decode(torch.tensor(z).float().to(device))
    for yidx in range(X.shape[0]):
        for xidx in range(X.shape[1]):
            zidx = yidx*X.shape[1] + xidx
            arch_val = decoded[zidx][input_idx]
            Z[yidx][xidx] = arch_val

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(np.array(L3), np.array(L4), np.array(Z), 60, cmap='binary')
    #ax.set_xlabel('l3')
    #ax.set_ylabel('l4')
    #ax.set_zlabel(input_name)

    # ax.view_init(45, 35)
    ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), cmap='viridis', edgecolor='none')
    plt.savefig('contour_l3_l4.png', bbox_inches='tight')
    
    # print("l1_valid:", l1_valid)
    # print("l2_valid:", l2_valid)
    # print("z_valid:", z_valid)

    exit(0)

    # Decode and evaluate
    output_name = "real_energy"
    perf_idx = 1

    reuse = False
    eval_cutoff = 2e11 # Don't visualize points that have evaluated cycle count/energy values greater than this
    if reuse:
        # Inspect decoded and evaluated values
        with open("l1_valid.pkl", "rb") as f:
            l1_valid = pickle.load(f)
        with open("l2_valid.pkl", "rb") as f:
            l2_valid = pickle.load(f)
        with open("cycle_valid.pkl", "rb") as f:
            cycle_valid = pickle.load(f)
        with open("energy_valid.pkl", "rb") as f:
            energy_valid = pickle.load(f)
        with open("area_valid.pkl", "rb") as f:
            area_valid = pickle.load(f)
    else:
        L1 = np.linspace(-plot_range, plot_range, 20) # Latent dim 1
        L2 = np.linspace(-plot_range, plot_range, 20) # Latent dim 2
        X, Y = np.meshgrid(L1, L2)  # Grid for contour3D purposes
        z = [None] * (X.shape[0] * X.shape[1]) # Holds latent vectors to decode

        # [16, 1024, 1024, 64, 16384, 1024, 65536] -> [0.1280, -0.4072,  1.3486,  1.1342]
        for yidx in range(X.shape[0]):
            for xidx in range(X.shape[1]):
                zidx = yidx*X.shape[1] + xidx
                #z[zidx] = [ X[yidx][xidx],  Y[yidx][xidx] ]# , 0, 0]
                z[zidx] = [ X[yidx][xidx],  Y[yidx][xidx] , 0, 0]
                # z = torch.tensor([[ X[yidx][xidx],  Y[yidx][xidx], 0, 0]]).double()
        
        # Get new arch
        decoded = model.decode(torch.tensor(z).float())

        l1_valid = list()
        l2_valid = list()
        cycle_valid = list()
        energy_valid = list()
        area_valid = list()
        for yidx in range(X.shape[0]):
            for xidx in range(X.shape[1]):
                zidx = yidx*X.shape[1] + xidx
                arch_config = decoded[zidx]

                print("Latent point:", z[zidx])
                print("Decoded arch:", arch_config)

                eval_result = eval_arch(arch_config)
                if eval_result is None:
                    continue
                cycle = eval_result[0]
                energy = eval_result[1]
                area = eval_result[2]

                # If only using valid outputs
                l1_valid.append(X[yidx][xidx])
                l2_valid.append(Y[yidx][xidx])
                cycle_valid.append(cycle)
                energy_valid.append(energy)
                area_valid.append(area)

        # print(Z[X.shape[0]-1][X.shape[1]-1])

        # print(Z[0][0])
        # exit(0)

        # arch_val = decoded[0][input_idx]
        # Z[yidx][xidx] = arch_val

        print("Number of valid points:", len(l1_valid))
        with open("l1_valid.pkl", "wb") as f:
            pickle.dump(l1_valid, f)
        with open("l2_valid.pkl", "wb") as f:
            pickle.dump(l2_valid, f)
        with open("cycle_valid.pkl", "wb") as f:
            pickle.dump(cycle_valid, f)
        with open("energy_valid.pkl", "wb") as f:
            pickle.dump(energy_valid, f)
        with open("area_valid.pkl", "wb") as f:
            pickle.dump(area_valid, f)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(np.array(L1), np.array(L2), np.array(Z), 60, cmap='binary')

    z_valid = None
    if perf_idx == 0:
        z_valid = cycle_valid
    elif perf_idx == 1:
        z_valid = energy_valid
    elif perf_idx == 2:
        z_valid = area_valid

    mask = np.array(z_valid) < eval_cutoff
    l1_valid = np.array(l1_valid)[mask]
    l2_valid = np.array(l2_valid)[mask]
    z_valid  = np.array(z_valid)[mask]

    ax.scatter(l1_valid, l2_valid, z_valid, c=z_valid, cmap='viridis', linewidth=0.5)

    ax.set_xlabel('l1')
    ax.set_ylabel('l2')
    ax.set_zlabel(output_name)

    # ax.view_init(25, 35)
    plt.savefig('contour_l1_l2_eval_scatter.png', bbox_inches='tight')

    ax.plot_trisurf(l1_valid, l2_valid, z_valid, cmap='viridis', edgecolor='none')
    plt.savefig('contour_l1_l2_eval_trisurf.png', bbox_inches='tight')


def random_search():
    random.seed(6)
    max = [64, 32, 256, 256, 4096, 256]
    scale = [1, 1, 1, 2**8, 1, 2**10]
    print(f"Min: {min}")
    print(f"Max: {max}")
    print(f"Scale: {scale}")

    best_cycle = float("inf")
    best_cycle_arch = None
    best_energy = float("inf")
    best_energy_arch = None
    best_edp = float("inf")
    best_edp_arch = None
    num_tested = 0
    success = 0
    num_to_test = 3000
    while success < num_to_test:
        num_tested += 1
        arch_config = []
        for i in range(len(max)):
            val = random.randint(1, max[i]) * scale[i]
            arch_config.append(val)
        eval_result = eval_arch(arch_config)
        if eval_result is None:
            print("Invalid arch:", arch_config)
            continue
        
        cycle = eval_result[0]
        energy = eval_result[1]
        edp = cycle * energy

        print("Arch evaluated:", arch_config)
        print(f"Cycle: {cycle}, Energy: {energy}, EDP: {edp}")
        success += 1

        if cycle < best_cycle:
            best_cycle = cycle
            best_cycle_arch = arch_config
        if energy < best_energy:
            best_energy = energy
            best_energy_arch = arch_config
        if edp < best_edp:
            best_edp = edp
            best_edp_arch = arch_config

        if num_tested == num_to_test:
            print(f"Best cycle count: {best_cycle}, Arch: {best_cycle_arch}")
            print(f"Best energy: {best_energy}, Arch: {best_energy_arch}")
            print(f"Best EDP: {best_edp}, Arch: {best_edp_arch}")
            print(f"{success} valid arch out of {num_tested} randomly searched")

    print(f"Best cycle count: {best_cycle}, Arch: {best_cycle_arch}")
    print(f"Best energy: {best_energy}, Arch: {best_energy_arch}")
    print(f"Best EDP: {best_edp}, Arch: {best_edp_arch}")
    print(f"{success} valid arch out of {num_tested} randomly searched")

def motivation():
    # simba = [16, 8, 64, 16384, 1024, 65536]
    simba = [16, 8, 64, 16384, 1024, 65536*4]
    arith_idx = 1
    # buf_idx = 2 # accbuf
    buf_idx = 3 # weight buf
    buf_instances = 128
    globalbuf_idx = 5
    # arith_options = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    arith_options = [4, 8, 12, 16, 20, 24, 28, 32]
    # buf_options = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384] # weightbuf options
    # buf_options = [11264, 12288, 13312, 14336, 15360, 16384, 17408]
    buf_options = [14848, 15872, 16896, 17920]
    # buf_options = [4, 8, 12, 16, 20, 24, 28, 32] # accbuf options

    for i in range(len(arith_options)):
        for j in range(len(buf_options)):
            arch_config = simba[:]
            arch_config[arith_idx] = arith_options[i]

            buf_entries_default = simba[buf_idx]
            buf_entries = buf_options[j]
            arch_config[buf_idx] = buf_entries

            globalbuf_entries_default = simba[globalbuf_idx]
            globalbuf_entries_new = globalbuf_entries_default + (buf_instances * (buf_entries_default - buf_entries))
            arch_config[globalbuf_idx] = globalbuf_entries_new

            eval_result = eval_arch(arch_config)
            if eval_result is None:
                print("Invalid arch:", arch_config)
                continue
            
            cycle = eval_result[0]
            energy = eval_result[1]
            edp = cycle * energy
            print("Arch evaluated:", arch_config)
            print(f"Cycle: {cycle}, Energy: {energy}, EDP: {edp}")


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y, energy) in enumerate(tqdm(data)):
        # copy igraph
        # otherwise original igraphs will save the H states and consume more GPU memory
        g_batch.append(g.tolist())
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            # g_batch = model._collate_fn(g_batch)
            g_batch = torch.Tensor(g_batch).to(device)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )


def interpolation_exp(epoch, num=5):
    print('Interpolation experiments between two random testing graphs')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    if args.data_type == 'BN':
        eva = Eval_BN(interpolation_res_dir)
    interpolate_number = 10
    model.eval()
    cnt = 0
    # Interpolate each pair of test points
    # for i in range(0, len(test_data), 2):
    for i in range(0, 2, 2):
        cnt += 1
        # Get two test points
        d0, d1 = test_data[i], test_data[i+1]
        # Convert 1d to 2d, and encode into latent space
        z0, _ = model.encode(d0[0].unsqueeze(0))
        z1, _ = model.encode(d1[0].unsqueeze(0))
        print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
        print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
        Z = []  # to store all the interpolation points
        for j in range(0, interpolate_number + 1):
            zj = z0 + (z1 - z0) / interpolate_number * j
            Z.append(zj)
        Z = torch.cat(Z, 0)

        # Decode from latent space
        arch_configs = model.decode(Z)
        names = []
        eval_results = []
        l1_vals = []
        l2_vals = []
        # Evaluate decoded points
        for j in range(0, interpolate_number + 1):
            # namej = 'graph_interpolate_{}_{}_of_{}'.format(i, j, interpolate_number)
            # namej = plot_config(G[j], interpolation_res_dir, namej, backbone=True, 
            #                  data_type=args.data_type)

            arch_config = arch_configs[j]

            eval_result = eval_arch(arch_config)
            if eval_result is None:
                print("Invalid arch:", arch_config)
                continue
            
            l1_vals.append(Z[j][0].item())
            l2_vals.append(Z[j][1].item())
            names.append(arch_config)
            eval_results.append(eval_result)
        # fig = plt.figure(figsize=(120, 20))

        print(names)
        print(eval_results)
        eval_idx = 0 # 0 for cycles, 1 for energy, 2 for area
        if eval_idx == 0:
            eval_name = "eval_cycle"
        elif eval_idx == 1:
            eval_name = "eval_energy"
        elif eval_idx == 2:
            eval_name = "eval_area"
        else:
            eval_name = "check eval_idx"

        # l1_vals = Z.detach().numpy()[:, 0]
        # l2_vals = Z.detach().numpy()[:, 1]
        z_vals = [result[0] for result in eval_results]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        ax.set_zlabel(eval_name)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        # ax.view_init(25, 35)
        ax.scatter(l1_vals, l2_vals, z_vals, c=z_vals, cmap='viridis', linewidth=0.5)
        plt.savefig('interp_eval_scatter.png', bbox_inches='tight')
        ax.plot(l1_vals, l2_vals, z_vals, linewidth=0.5)
        plt.savefig('interp_eval_line.png', bbox_inches='tight')
        
        arch_idx = 5
        arch_name = get_arch_name(arch_idx)

        z_vals = [name[arch_idx].item() for name in names]

        ax = plt.axes(projection='3d')
        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        ax.set_zlabel(arch_name)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.scatter(l1_vals, l2_vals, z_vals, c=z_vals, cmap='viridis', linewidth=0.5)
        plt.savefig('interp_arch_scatter.png', bbox_inches='tight')
        ax.plot(l1_vals, l2_vals, z_vals, linewidth=0.5)
        plt.savefig('interp_arch_line.png', bbox_inches='tight')

        scale = [2, 2, 64, 32, 256, 65536, 4096, 262144, 10000000, 10000000000]

        import seaborn as sns
        import pandas as pd
        for idx in range(len(names)):
            l1 = l1_vals[idx]
            l2 = l2_vals[idx]
            arch_vals = [arch_val.item() for arch_val in names[idx]]
            eval_vals = list(eval_results[idx])
            # fig = plt.figure()
            # ax = fig.add_subplot()
            # ax2 = ax.twinx()
            # ax3 = ax.twinx()
            # ax4 = ax.twinx()
            # ax5 = ax.twinx()
            # ax6 = ax.twinx()
            # ax7 = ax.twinx()
            # ax8 = ax.twinx()
            # width = 0.1
            # ax.barh([arch_vals[0]], label=get_arch_name(0), width=width)
            # ax2.barh([arch_vals[1]], label=get_arch_name(1), width=width)
            # ax3.barh([arch_vals[2]], label=get_arch_name(2), width=width)
            # ax4.barh([arch_vals[3]], label=get_arch_name(3), width=width)
            # ax5.barh([arch_vals[4]], label=get_arch_name(4), width=width)
            # ax6.barh([arch_vals[5]], label=get_arch_name(5), width=width)
            # ax7.barh([eval_vals[0]], label="eval_cycle", width=width)
            # ax8.barh([eval_vals[1]], label="eval_energy", width=width)
            # plt.savefig(f'interp_{str(z0)}_to_{str(z1)}_{idx}.png', bbox_inches='tight')

            raw_vals = [l1, l2] + arch_vals + eval_vals[:2]
            normalized = [raw_vals[k] / scale[k] for k in range(len(raw_vals))]
            d = {
                "idx": [str(idx)] * 10,
                "norm_val": normalized,
                "name": ["l1", "l2"] + [get_arch_name(i) for i in range(6)] + ["eval_cycle", "eval_energy"]
            }
            df = pd.DataFrame(data=d)
            fig = plt.figure()
            ax = sns.barplot(y="idx", x="norm_val", hue="name", data=df)
            fig.savefig(f'interp_{str(z0)}_to_{str(z1)}_{idx}.png', bbox_inches='tight')

        # Visualize
        # for j, namej in enumerate(names):
        #     imgj = mpimg.imread(namej)
        #     fig.add_subplot(1, interpolate_number + 1, j + 1)
        #     plt.imshow(imgj)
        #     if args.data_type == 'BN':
        #         plt.title('{}'.format(scores[j]), fontsize=40)
        #     plt.axis('off')
        # plt.savefig(os.path.join(args.res_dir, 
        #             args.data_name + '_{}_interpolate_exp_ensemble_epoch{}_{}.pdf'.format(
        #             args.model, epoch, i)), bbox_inches='tight')
        '''
        # draw figures with the same height
        new_name = os.path.join(args.res_dir, args.data_name + 
                                '_{}_interpolate_exp_ensemble_{}.pdf'.format(args.model, i))
        combine_figs_horizontally(names, new_name)
        '''
        if cnt == num:
            break


def interpolation_exp2(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments between flat-net and dense-net')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation2')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 10
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    g1 = [[1]+[1]*(i-1) for i in range(1, n-1)]  # this is dense-net

    g0, _ = decode_ENAS_to_igraph(str(g0))
    g1, _ = decode_ENAS_to_igraph(str(g1))
    z0, _ = model.encode(g0)
    z1, _ = model.encode(g1)
    print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        zj = z0 + (z1 - z0) / interpolate_number * j
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)  
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_config(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    fig = plt.figure(figsize=(120, 20))
    for j, namej in enumerate(names):
        imgj = mpimg.imread(namej)
        fig.add_subplot(1, interpolate_number + 1, j + 1)
        plt.imshow(imgj)
        plt.axis('off')
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_interpolate_exp2_ensemble_epoch{}.pdf'.format(
                args.model, epoch)), bbox_inches='tight')


def interpolation_exp3(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments around a great circle')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation3')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 36
    model.eval()
    n = graph_args.max_n
    g0 = [[1]+[0]*(i-1) for i in range(1, n-1)]  # this is flat-net
    g0, _ = decode_ENAS_to_igraph(str(g0))
    z0, _ = model.encode(g0)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    # there are infinite possible directions that are orthogonal to z0,
    # we just randomly pick one from a finite set
    dim_to_change = random.randint(0, z0.shape[1]-1)  # this to get different great circles
    print(dim_to_change)
    z1[0, dim_to_change] = -(z0[0, :].sum() - z0[0, dim_to_change]) / z0[0, dim_to_change]
    z1 = z1 / torch.norm(z1) * norm0
    print('z0: ', z0, 'z1: ', z1, 'dot product: ', (z0 * z1).sum().item())
    print('norm of z0: {}, norm of z1: {}'.format(norm0, torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
    omega = torch.acos(torch.dot(z0.flatten(), z1.flatten()))
    print('angle between z0 and z1: {}'.format(omega))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        theta = 2*math.pi / interpolate_number * j
        zj = z0 * np.cos(theta) + z1 * np.sin(theta)
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type) 
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_config(G[j], interpolation_res_dir, namej, backbone=True, 
                         data_type=args.data_type)
        names.append(namej)
    # draw figures with the same height
    new_name = os.path.join(args.res_dir, args.data_name + 
                            '_{}_interpolate_exp3_ensemble_epoch{}.pdf'.format(args.model, epoch))
    combine_figs_horizontally(names, new_name)


def smooth_exp1(epoch):
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    save_dir = os.path.join(args.res_dir, 'smoothness')
    data = loadmat(latent_mat_name)
    # X_train, Y_train = extract_latent(train_data)
    X_train = data['Z_train']
    torch.manual_seed(123)
    # z0 = torch.tensor(torch.randn(1, model.nz), device='cuda:0', requires_grad=False)
    z0 = torch.tensor(torch.randn(1, model.nz), requires_grad=False)
    print(z0)
    model.predictor(z0)
    max_xy = 0.3
    #max_xy = 0.6
    x_range = np.arange(-max_xy, max_xy, 0.005)
    y_range = np.arange(max_xy, -max_xy, -0.005)
    n = len(x_range)
    x_range, y_range = np.meshgrid(x_range, y_range)
    x_range, y_range = x_range.reshape((-1, 1)), y_range.reshape((-1, 1))   
    if True:  # select two principal components to visualize
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X_train)
        d1, d2 = pca.components_[0:1], pca.components_[1:2]
        print("PCA components:")
        print(d1, d2)
        new_x_range = x_range * d1
        new_y_range = y_range * d2
        grid_inputs = torch.FloatTensor(new_x_range + new_y_range) #.cuda()
    else:
        grid_inputs = torch.FloatTensor(np.concatenate([x_range, y_range], 1)) #.cuda()
        if args.nz > 2:
                grid_inputs = torch.cat([grid_inputs, z0[:, 2:].expand(grid_inputs.shape[0], -1)], 1)
    valid_arcs_grid = []
    batch = 3000
    max_n = graph_args.max_n
    data_type = args.data_type
    for i in range(0, grid_inputs.shape[0], batch):
        batch_grid_inputs = grid_inputs[i:i+batch, :]
        # valid_arcs_grid += decode_from_latent_space(batch_grid_inputs, model, 10, max_n, False, data_type) 
        valid_arcs_grid += batch_grid_inputs
    
    print("Evaluating 2D grid points")
    print("Total points: " + str(grid_inputs.shape[0]))
    grid_scores = []
    x, y = [], []
    load_module_state(model, os.path.join(args.res_dir, 
                                            'model_checkpoint{}.pth'.format(300)))
    load_module_state(optimizer, os.path.join(args.res_dir, 
                                            'optimizer_checkpoint{}.pth'.format(300)))
    load_module_state(scheduler, os.path.join(args.res_dir, 
                                            'scheduler_checkpoint{}.pth'.format(300)))
    for i in range(len(valid_arcs_grid)):
        arc = valid_arcs_grid[ i ] 
        if arc is not None:
            score = model.predictor(arc)[0].cpu().detach().numpy()
            x.append(x_range[i, 0])
            y.append(y_range[i, 0])
            grid_scores.append(score)
        else:
            score = 0
            print(i)

    vmin, vmax = np.min(grid_scores), np.max(grid_scores)
    ticks = np.linspace(vmin, vmax, 9, dtype=float).tolist()
    cmap = plt.cm.get_cmap('viridis')
    #f = plt.imshow(grid_y, cmap=cmap, interpolation='nearest')
    sc = plt.scatter(x, y, c=grid_scores, cmap=cmap, vmin=vmin, vmax=vmax, s=10)
    plt.colorbar(sc, ticks=ticks)
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(save_dir + "2D_vis_{}.pdf".format(epoch))
    plt.savefig(save_dir + "2D_vis_{}.png".format(epoch))

def pred_arch_perf(dnn_def_path, model, device, log_layerfeat=False, norm_layerfeat=False, norm_layerfeat_option='', norm_latent=False, log_obj=False, norm_obj=False, norm_path=None):
    # perf_type : "cycle", "energy", or "edp"
    # input_idx : Doesn't matter if plotting predicted perf, specify otherwise

    ############## Test encoding 
    real_img_manual = torch.tensor([
        [16, 32, 208, 1536, 1888, 84992],
        [32, 32, 256, 51712, 1950, 110592],
        # 32_32_254_51712_1950_110592 Best 10
        [16, 30, 160, 53760, 624, 189440],
        # 16_30_160_53760_624_189440 Avg
        [32, 10, 142, 35072, 1860, 1024],
        # 32_10_142_35072_1860_1024     
        [64, 32, 118, 8192, 3318, 95232],
        [64, 24, 62, 18126, 1664, 166770],
    ]).to(device)
    #], dtype=torch.float64).to(device)
    cycle_manual = [
        28224,
        28224,
        47335,
        522650,
        28224,
        37632,
    ]
    energy_manual = [
        95272500,
        9.99E+07,
        2.41E+08,
        7.07E+08,
        99949980.94,
        117524767.1,
    ] 

    edp_manual = [
        latency * energy_manual[i]
        for i, latency in enumerate(cycle_manual)
           ]

    mu, logvar = model.encode(real_img_manual)
    # z = model.reparameterize(mu, logvar)
        
    print(f"mu {mu}")
    print(f"logvar {logvar}")
    # print(f"z: {z}")

    decoded = model.decode(mu)
    print(f"decoded: {decoded}")

    dnn_def_tensor, num_predictors = parse_dnn_def(dnn_def_path, device, log_layerfeat, norm_layerfeat, norm_layerfeat_option, norm_path=norm_path)
    # dnn_def_tensor = dnn_def_tensor.repeat(z.size) 

    # generate predictors for inference
    gen_dnn_predictors(model, num_predictors)

    print("Log energy:")
    print(np.log(energy_manual))
    print("Normalized energy:")
    print((np.log(energy_manual) - 18.923) / 0.2390)

    cycles = []
    energies = []
    print("num data: ", mu.size()[0])
    for i in range(mu.size()[0]):
        if norm_latent: 
            cycle, energy = model.dnn_perf(mu[i]/10, dnn_def_tensor) 
        else:
            cycle, energy = model.dnn_perf(mu[i], dnn_def_tensor) 
        pred_cycle = cycle[0].item() # * 2**28 # * 10**-6
        pred_energy = energy[0].item() #  * 2**38 # * 10**-9
        print(pred_cycle, pred_energy)
        
        pred_cycle, pred_energy = denorm_obj(pred_cycle, pred_energy, log_obj, norm_obj, norm_path) 

        cycles.append(pred_cycle)
        energies.append(pred_energy)

    # cycle_manual = pred_cycle
    # energy_manual = pred_energy
    print(f"real cycles: {cycle_manual}")
    print(f"pred cycles: {cycles}")
    print(f"real energy: {energy_manual}")
    print(f"pred energy: {energies}")
    print(f"real edp: {edp_manual}")
    edp = [latency * energies[i] for i, latency in enumerate(cycles)] 
    print(f"pred edp: {edp}")

    get_percent_diff(cycle_manual, cycles, 'latency')
    get_percent_diff(energy_manual, energies, 'energy')
    get_percent_diff(edp_manual, edp, 'edp')

def pred_vis(dnn_def_path, model, device, nz, save_to="png", log_layerfeat=False, norm_layerfeat=False, norm_layerfeat_option='', norm_latent=False, log_obj=False, norm_obj=False, norm_path=None):
    # perf_type : "cycle", "energy", or "edp"
    # input_idx : Doesn't matter if plotting predicted perf, specify otherwise
    plot_range = (-4, 4)
    L1 = np.linspace(*plot_range, 40) # Latent dim 1
    L2 = np.linspace(*plot_range, 40) # Latent dim 2
    X, Y = np.meshgrid(L1, L2)  # Grid for contour3D purposes
    z = [None] * (X.shape[0] * X.shape[1]) # Holds latent vectors to decode

    # [16, 1024, 1024, 64, 16384, 1024, 65536] -> [0.1280, -0.4072,  1.3486,  1.1342]
    for yidx in range(X.shape[0]):
        for xidx in range(X.shape[1]):
            zidx = yidx*X.shape[1] + xidx
            point = [0] * nz
            # point = list(random_best_points[0].detach())
            point[0] = X[yidx][xidx]
            point[1] = Y[yidx][xidx]
            z[zidx] = point
            # z[zidx] = [ X[yidx][xidx],  Y[yidx][xidx], 0, 0]
    z = torch.tensor(z)
    decoded = model.decode(z.float().to(device))

    dnn_def_tensor, num_predictors = parse_dnn_def(dnn_def_path, device, log_layerfeat, norm_layerfeat, norm_layerfeat_option, norm_path=norm_path)

    # generate predictors for inference
    gen_dnn_predictors(model, num_predictors)

    if norm_latent: 
        cycles, energies = model.dnn_perf(z/10, dnn_def_tensor, batched=True) 
    else:
        cycles, energies = model.dnn_perf(z, dnn_def_tensor, batched=True) 
    cycles = cycles.detach().numpy()
    energies = energies.detach().numpy()

    cycles, energies = denorm_obj(cycles, energies, log_obj, norm_obj, norm_path) 

    print(energies)
    cycles = cycles.reshape(X.shape) * 10**-6
    energies = energies.reshape(X.shape) * 10**-9
    
    layer_file_name = dnn_def_path.split("/")[-1].split(".")[0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.view_init(30, 30)
    ax.plot_trisurf(X.flatten(), Y.flatten(), cycles.flatten(), cmap='viridis', alpha=1.0)
    ax.set_xlabel('Latent Dim. 1')
    ax.set_ylabel('Latent Dim. 2')
    ax.set_zlabel('Latency (MCycles)')
    save_file = f'vis/{layer_file_name}_pred_cycle.{save_to}'
    plt.savefig(save_file, bbox_inches='tight')
    print(f"Saved contour plot of cycle prediction to", save_file)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.view_init(90, 270)
    ax.plot_trisurf(X.flatten(), Y.flatten(), energies.flatten(), cmap='viridis', alpha=1.0)
    ax.set_xlabel('Latent Dim. 1')
    ax.set_ylabel('Latent Dim. 2')
    ax.set_zlabel('Energy (J)')
    # ax.set_zlim((0.16, 0.17))
    save_file = f'vis/{layer_file_name}_pred_energy.{save_to}'
    plt.savefig(save_file, bbox_inches='tight')
    print(f"Saved contour plot of energy prediction to", save_file)

def encoded_data(args, train_data, model, num_points):
    ############## View encoded values
    data = train_data
    pbar = tqdm(data)
    g_batch = []
    latent_points = None
    for i, (g, cycle, energy, layer_feats) in enumerate(pbar):
        g_batch.append(g.tolist())
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = torch.Tensor(g_batch)
            mu, logvar = model.encode(g_batch)
            if latent_points is None:
                latent_points = mu.detach().numpy()
            else:
                latent_points = np.concatenate((latent_points, mu.detach().numpy()), axis=0)
            g_batch = []
            if len(latent_points) >= num_points:
                break
    return latent_points

def encoded_vis(args, train_data, model, latent_points, plot_color="viridis", plot_range=None, perf_type="cycle", arch_idx=0, save_to="png", log_obj=False, norm_obj=False, norm_path=None):
    # plot_color : e.g. "viridis", "Blues", "Oranges"
    # perf_type : "cycle", "energy", "edp"
    # arch_idx : 0 - 5
    
    # for i in range(len(latent_points)):
    #     point = latent_points[i]
    #     input_data = data[i]
    #     if any(np.abs(point) > 4):
    #         print("input:", input_data)
    #         print("z:", point)

    l1 = latent_points[:,0]
    if args.nz >= 2:
        l2 = latent_points[:,1]
    if args.nz >= 3:
        l3 = latent_points[:,2]
    if args.nz >= 4:
        l4 = latent_points[:,3]

    # fig = plt.figure()
    # ax = plt.axes()
    # ax.scatter(l1, l2, linewidth=0.1)
    # ax.set_xlabel('l1')
    # ax.set_ylabel('l2')
    # ax.set_zlabel('l3')
    # ax.view_init(0, 0)
    # plt.savefig('test_data_encoded.png', bbox_inches='tight')

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
        if n == -1:
            n = cmap.N
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cycles = train_data.arch_feats_frame["unique_cycle_sum"][:len(latent_points)]
    energies = train_data.arch_feats_frame["unique_energy_sum"][:len(latent_points)]
    cycles, energies = denorm_obj(cycles, energies, log_obj, norm_obj, norm_path) 
    edp = cycles * energies

    name = get_perf_name(perf_type)
    if perf_type == "cycle":
        z = cycles * 10**-6
    elif perf_type == "energy":
        z = energies * 10**-9
    elif perf_type == "edp":
        z = edp * 10**-15
    else:
        z = train_data.arch_feats_frame[get_arch_feat_name(arch_idx)][:len(latent_points)]# / 1024
        if arch_idx == 2:
            z = z * 384 / 1024 # each AccBuf entry adds 384 B of size
        elif arch_idx == 3:
            z = z * 128 / 1024 # each WeightBuf entry adds 128 B of size
        elif arch_idx == 4:
            z = z * 64 / 1024 # each InputBuf entry adds 64 B of size
        elif arch_idx == 5:
            z = z / 1024 # each GlobalBuf entry adds 1 B of size
        name = get_arch_name_pretty(arch_idx)
    name_cleaned = name.replace(" ", "_")
    name_cleaned = name_cleaned.replace("#", "Num")
    name_cleaned = name_cleaned.replace("*", "x")

    cutoff_percentile = 0.8
    cutoff_num = int(len(edp) * cutoff_percentile) - 1
    cutoff = np.partition(edp, cutoff_num)[cutoff_num]
    mask = edp < cutoff
    l1 = np.array(l1)[mask]
    l2 = np.array(l2)[mask]
    if args.nz >= 4:
        l3 = np.array(l3)[mask]
        l4 = np.array(l4)[mask]
    z  = np.array(z)[mask]

    if plot_color == "viridis":
        cmap = plt.get_cmap("viridis")
    else:
        cmap = truncate_colormap(plt.get_cmap(plot_color), 0.15, 0.9)
    
    # norm = plt.Normalize(0, 20)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scatter = ax.scatter(l1, l2, l3, c=z, s=(l4 - min(l4)+0.5)*5, cmap=cmap, alpha=0.7)#, norm=norm)
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label(name)
    ax.set_xlabel('Latent Dim. 1')
    ax.set_ylabel('Latent Dim. 2')
    ax.set_zlabel('Latent Dim. 3')
    ax.view_init(30, 60)
    if plot_range:
        ax.set_xlim(*plot_range)
        ax.set_ylim(*plot_range)
    save_file = 'vis/train_data_encoded_3d.' + save_to
    plt.savefig(save_file, bbox_inches='tight')
    print("Saved encoded training data (dim 1, 2, 3, 4) to", save_file)

    scatter_args = {
        "s": 10,
        "alpha": 0.7
    }
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(l1, l2, c=z, linewidth=0.1, cmap=cmap, **scatter_args)
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label(name)
    ax.set_xlabel('Latent Dim. 1')
    ax.set_ylabel('Latent Dim. 2')
    save_file = f'vis/train_data_encoded_{name_cleaned}_l1l2.' + save_to
    plt.savefig(save_file, bbox_inches='tight')
    print("Saved encoded training data (dim 1 and 2) to", save_file)

    if args.nz >= 4:
        fig = plt.figure()
        ax = plt.axes()
        ax.scatter(l3, l4, c=z, linewidth=0.1, cmap=cmap, **scatter_args)
        cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
        cbar.set_label(name)
        ax.set_xlabel('Latent Dim. 3')
        ax.set_ylabel('Latent Dim. 4')
        save_file = f'vis/train_data_encoded_{name_cleaned}_l3l4.' + save_to
        plt.savefig(save_file, bbox_inches='tight')
        print("Saved encoded training data (dim 3 and 4) to", save_file)

