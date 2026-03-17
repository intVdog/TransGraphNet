def generate_large_ddos_dot(filename="ddos_large.dot", num_bots=85):
    with open(filename, "w") as f:
        f.write("digraph G {\n")

        node_id = 1
        targets = [
            {"name": "webServer", "svc": "http", "port": "80", "type": "tcp"},
            {"name": "dnsServer", "svc": "dns", "port": "53", "type": "udp"},
            {"name": "mailServer", "svc": "smtp", "port": "25", "type": "tcp"},
            {"name": "dbServer", "svc": "mysql", "port": "3306", "type": "tcp"},
            {"name": "fileServer", "svc": "smb", "port": "445", "type": "tcp"}
        ]

        # 1. Generate attack sources (Botnets) - about 85 nodes
        bot_ids = []
        for i in range(1, num_bots + 1):
            f.write(f'  {node_id} [label="{node_id}:attackerLocated(botnet{i}):1",shape=box];\n')
            bot_ids.append(node_id)
            node_id += 1

        # 2. Generate network access control (HACL) - 85 corresponding nodes
        hacl_ids = []
        for i, b_id in enumerate(bot_ids):
            target = targets[i % len(targets)]
            f.write(
                f'  {node_id} [label="{node_id}:hacl(botnet{i + 1},{target["name"]},{target["type"]},{target["port"]}):1",shape=box];\n')
            hacl_ids.append(node_id)
            node_id += 1

        # 3. Generate attack logic chains for each target (7 nodes per target, 35 total)
        for target in targets:
            # Service Info
            svc_info_id = node_id
            f.write(
                f'  {svc_info_id} [label="{svc_info_id}:networkServiceInfo({target["name"]},{target["svc"]},{target["type"]},{target["port"]},version1):1",shape=box];\n')
            node_id += 1

            # Rule 6 & NetAccess
            r6_id = node_id
            f.write(f'  {r6_id} [label="{r6_id}:RULE 6 (direct network access):0",shape=ellipse];\n')
            node_id += 1

            na_id = node_id
            f.write(
                f'  {na_id} [label="{na_id}:netAccess({target["name"]},{target["type"]},{target["port"]}):0",shape=diamond];\n')
            node_id += 1

            # Rule 10 & DDoS Traffic
            r10_id = node_id
            f.write(f'  {r10_id} [label="{r10_id}:RULE 10 (traffic flooding):0",shape=ellipse];\n')
            node_id += 1

            ddos_id = node_id
            f.write(f'  {ddos_id} [label="{ddos_id}:ddosTraffic({target["name"]}):0",shape=diamond];\n')
            node_id += 1

            # Rule 12 & Service Overload
            r12_id = node_id
            f.write(f'  {r12_id} [label="{r12_id}:RULE 12 (resource exhaustion):0",shape=ellipse];\n')
            node_id += 1

            overload_id = node_id
            f.write(f'  {overload_id} [label="{overload_id}:serviceOverload({target["name"]}):0",shape=diamond];\n')
            node_id += 1

            # --- Connection Logic ---
            # Connect corresponding Botnet and HACL to Rule 6
            target_bots = [bot_ids[i] for i in range(len(bot_ids)) if i % len(targets) == targets.index(target)]
            target_hacls = [hacl_ids[i] for i in range(len(hacl_ids)) if i % len(targets) == targets.index(target)]

            for b, h in zip(target_bots, target_hacls):
                f.write(f"  {b} -> {r6_id};\n")
                f.write(f"  {h} -> {r6_id};\n")

            f.write(f"  {r6_id} -> {na_id};\n")
            f.write(f"  {na_id} -> {r10_id};\n")
            f.write(f"  {svc_info_id} -> {r10_id};\n")
            f.write(f"  {r10_id} -> {ddos_id};\n")
            f.write(f"  {ddos_id} -> {r12_id};\n")
            f.write(f"  {r12_id} -> {overload_id};\n")

        f.write("}\n")


generate_large_ddos_dot()
print("Generated ddos_large.dot file with approximately 205 nodes.")