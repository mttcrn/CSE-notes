---
layout:
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# Computing Infrastructures

## HW infrastructures

## System level

A **computing infrastructure** is a technological infrastructure that provides hardware and software for computation to other systems and services. \
For example a server, that is mine and that I can access (and no one else) from different part of the world, is not a computing infrastructure. On the other hand, if I access a server in which some code is running, and some other people can access and use it, it is a computing infrastructure.

#### Data center

A data center is composed by 3 server: for processing, for storage and for communications.

<div align="center"><figure><img src="/assets/image (102).png" alt="" width="375"><figcaption></figcaption></figure></div>

To access the resources a **virtualization layer** is needed.

* **VMMs**: provide the full stack (OS, LIB, APP). Application depends on guest OS.
* **Containers**: applications are packaged with all their dependencies into a standardized unit for software development/deployment.

<div align="left"><figure><img src="/assets/image (103).png" alt="" width="375"><figcaption><p>Virtual Machine Manager</p></figcaption></figure> <figure><img src="/assets/image (104).png" alt="" width="375"><figcaption><p>Container</p></figcaption></figure></div>

#### Edge Computing

Edge computing aim at moving the computing outside the cloud (such as 5G technology).

<div data-full-width="false"><figure><img src="/assets/Untitled 3 (1) (1).png" alt="" width="188"><figcaption></figcaption></figure></div>

#### Embedded PCs

Embedded PCs are PC in which we just have the main board (such as Raspberry). It support the pervasive computation. They suffer from the fact that they are a computer.

<figure><img src="/assets/Untitled 4 (1) (1).png" alt="" width="302"><figcaption></figcaption></figure>

#### Internet of Things (IoT)

Internet-of-Things (IoT) is an environment in which the computation is divided into different devices.

<figure><img src="/assets/Untitled 5 (1) (1).png" alt="" width="339"><figcaption></figcaption></figure>

An IT perspective for computing infrastructures:\
The software is the reason why the system exists. The operating system “mask” the hardware level. The term “architecture” means that we are organizing spaces, in order for them to have a goal.

<div><figure><img src="/assets/Untitled 6 (1) (1).png" alt=""><figcaption></figcaption></figure> <figure><img src="/assets/Untitled 7 (1) (1).png" alt=""><figcaption></figcaption></figure></div>

### Data Centers (DC)

In the last few decades, computing and storage have moved from PC-like clients to smaller, often mobile, devices, combined with large internet services. Traditional enterprises are also shifting to cloud computing.

Advantages to the user:

* **User experience improvements**: ease of management (no configuration or backups needed) and ubiquity of access.

Advantages to the vendor:

* **SasS** (software as service) allows **faster application development** (easier to make changes and improvements).
* **Improvements** and **fixes** in the software are easier inside their data centers (instead of updating many millions of clients with peculiar hardware and software configurations).
* The **hardware deployment** is restricted to a few well-tested configurations.

Advantages to the owner of the CI (server-side computing allow):

* **Faster introduction** of **new hardware** devices.
* Many application services can run at a **low cost per user**.

Some workloads require so much computing capability that they are a more natural fit in data center (and not in client-side computing). For examples search services (like web or images) or machine and deep learning.

#### Warehouse-scale computers vs. datacenters

The trends toward server-side computing and widespread internet services created a new class of computing systems, the **warehouse-scale computers** (**WSCs**). A **massive scale** of the software infrastructure, data repositories, and hardware platform. The program is an **internet service**. It may consist of tens or more individual programs, such programs interact to **implement complex end-user services** such as email, search, maps or machine learning.

Data centers are buildings where **multiple servers** and **communication units** are **co-located** because of their common environmental requirements and physical security needs, and for ease of maintenance. A traditional data centers (typically):

* Hosts a large number of relatively small/medium sized application.
* Each application is running on a dedicated hardware infrastructure that is de-coupled and protected from other systems in the same facility.
* Applications tend not to communicate with each other.

Those data centers host hardware and software for multiple organizational units or even different companies.

WSCs belong to a **single organization**, they use a relatively **homogeneous hardware** and **system software platform**, and share a **common system management layer** (such as google, facebook, amazon, dropbox, ..).

* WSCs run a **small number** of **very small application** (or internet of services) but in general to a very large number of users.
* The common resource management infrastructure allows significant **deployments flexibility**.
* The requirements of homogeneity, single-organization control and cost efficiency motivate the designer to take new approaches in designing WSCs.

<div><figure><img src="/assets/Untitled 10 (1) (1).png" alt=""><figcaption></figcaption></figure> <figure><img src="/assets/Untitled 9 (1) (1).png" alt=""><figcaption></figcaption></figure></div>

#### From data centers to warehouse-scale computer (and back)

WSCs were initially designed for online data-intensive web workloads. Now they are used also for public clouds computing systems (Amazon, Google, Microsoft, ..). Such public clouds do run many small applications, like a traditional data center. All of these applications rely on **VM** (or **Containers**), and they access large, common services for block or database storage, load balancing, and so on, fitting very well with the WSC model.

WSCs are not just a collection of servers: the software running on these systems executes on clusters of hundreds to thousands of individual servers (far beyond a single machine or a single rack). The machine is itself this large cluster or aggregation of servers and needs to be considered as a single computing unit.

Multiple datacenters are (often) **replicas of the same service**: to reduce user latency, to improve serving throughput. A request is typically fully processed within one data center.

The world is divided into **geographic areas** (GAs) defined by geo-political boundaries (or country borders), determined mainly by data residency, in each GA there are at least 2 computing regions.

**Computing regions** are seen by the customers as the **finer grain discretization** of the infrastructure. It is possibile to exchange data between data centers of the same computing region having a fixed latency. The region are far enough such that if there is a large blackout or atmospherical event, the infrastructure can survive.

<figure><img src="/assets/Untitled 11 (1) (1).png" alt="" width="264"><figcaption></figcaption></figure>

The **availability zone** (AZs) are **finer grain location** within a single computing region. They allow customers to run mission critical application with **high availability** and **fault tolerance** to **datacenters failures**. It is different from the concept of availability set. Application-level synchronous replication among AZs. 3 is the minimum number of AZs and enough for quorum.

<div><figure><img src="/assets/Untitled 12 (1) (1).png" alt=""><figcaption></figcaption></figure> <figure><img src="/assets/Untitled 13 (1).png" alt=""><figcaption></figcaption></figure></div>

**Services provided through WSCs** must guarantee **high availability**, typically aiming for at least **99.99% uptime**. Achieving such fault-free operation is difficult when a large collection of hardware and system software is involved. WSC workloads must be designed to gracefully tolerate large numbers of component faults with little or no impact on service level performance and availability. This is exactly the goal of the **dependability**.

#### Architectural overview of warehouse-scale computers

Hardware implementation of WCSs might differ significantly each other (different markets, scenarios, ..). However, the **architectural organization** of these systems is relatively **stable**.

<figure><img src="/assets/Untitled 14 (1).png" alt="" width="522"><figcaption></figcaption></figure>

**Servers**: the main processing equipment.

* Like ordinary PC, but with a form that allows to fit them into the racks (blade enclosure format, tower, ..)
* They may differ in number and type of CPUs, available RAM, locally attached disks (HDD, SDD or not installed), and other special purpose devices (like GPUs, DSPs, and co-processors).

**Storage**: such as disks and flash SSDs. These devices are connected to the data center network and managed by sophisticated distributed systems.

* **DAS**: direct attached storage.
* **NAS**: network attached storage.
* **SAN**: storage area network.
* **RAID** (redundant array of inexpensive disks) controllers.

<figure><img src="/assets/image (92).png" alt="" width="326"><figcaption></figcaption></figure>

**Networking**: providing internal and external connections, and allow networks interconnections among the devices. They can be **hubs**, **routers**, **DNS or DHCP servers**, **load balancers**, technology switches, **firewalls**, and many more. It masks the file system to the user. The data center is not effective without networking.

**Building and infrastructure**: there is a need for a comprehensive design of computation, storage, networking and building infrastructure. Data centers can be very large (up to set of buildings), they can consume a lot of power and, in general, need 99.99% uptime (one hour downtime per year). Everything must be correctly designed.

## Node level

## Server

Server are the **basic building blocks** of WSCs. They are **interconnected** by **hierarchies of networks**, and supported by the **shared power** and **cooling infrastructure**. We must guarantee that in case of failures we can recover the largest number of servers: so we have redundant power supply. They are typically hosted in individual shelves.

Servers are usually **build in** a **tray** or **blade enclosure** format, housing the motherboard, chipset and additional plug-in components.

The **motherboard** provides **sockets** and **plug-in slots** to install **CPUs**, **memory modules** (DIMMs, dual in-line memory modules), **local storage** (such as Flash SSDs or HDDs), and **network interface cards** (NICs) to satisfy the range of resource requirements. WSCs use a relatively homogeneous hardware and system software platform: in this way the price of each motherboard is typically cheaper since a lot of them are bought at the same time, and the maintenance is easier since all the motherboards are the same.

**Racks** are special **shelves** that accomodate all the IT equipment and allow their interconnection. Server racks are measures in rack units (**1U = 44.45 mm**).\
The advantage of using these is that it allows designers to stack up other electronic devices along with the servers. IT equipment must conform to specific sizes to fit into the rack shelves. The rack is the shelf that holds tens of servers together. It handles shared power delivery, battery backup, and power conversion. It is often convenient to connect the network cables at the top of the rack, such a rack-level switch is appropriately called a TOR (Top of Rack) switch.

<figure><img src="/assets/Untitled 16 (1).png" alt="" width="226"><figcaption></figcaption></figure>

* **Tower** servers: looks and fells much like a traditional tower PC.

<figure><img src="/assets/Untitled 17 (1).png" alt=""><figcaption></figcaption></figure>

* **Rack** servers: are designed to be positioned in a bay, by vertically stacking servers one over the other along with other devices (storage units, cooling systems, network peripherals, batteries).

<figure><img src="/assets/Untitled 18 (1).png" alt=""><figcaption></figcaption></figure>

* **Blade** servers: latest and most advanced type of servers in the market. They can be termed as **hybrid rack** servers, in which servers are placed inside blade enclosures, forming a blade system.

<figure><img src="/assets/Untitled 19 (1).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/Untitled 20.png" alt="" width="563"><figcaption></figcaption></figure>

#### From the rack to the datacenter

The IT equipment is stored into **corridors** (in order to **allow air flow** and **management**), and **organized into racks**. Cold air flows from the front (cool aisle), cools down the equipment, and leave the room from the back (warm aisle). Corridors where servers are located are split into cold aisle, where the front panels of the equipment is reachable, and warm aisle, where the back connections are located. A roof is putted between the front faces of the rack, in order not to throw away fresh air.

Cooling represent 25% or the entire costs of the data center management.

<div><figure><img src="/assets/image (105).png" alt=""><figcaption></figcaption></figure> <figure><img src="/assets/Untitled (2) (1).png" alt=""><figcaption></figcaption></figure></div>

There is a **mismatch**: the complexity increases faster than the performance of the current technology. In order to overcome this problem, the idea is to **parallelize** inside the same machine more effectively. To satisfy this growth, WSCs deploy specialized **accelerator hardware** such as **GPU**, **TPU** or **FPGA**.

<figure><img src="/assets/image (106).png" alt="" width="188"><figcaption></figcaption></figure>

In **GPU** (graphical processing unit) the same program is executed on many data elements in parallel (**data parallel computations**). The scientific codes are mapped onto the matrix operations. Up to 1000x faster than CPU.\
The performance is limited by the **slowest learner** (**slowest GPU**, since data can be aggregated when all GPUs have finished their computation) and **transmitting data** (that can be resolved with an high performance network).

GPUs are configured with a **CPU host connected to a PCle-attached accelerator tray with multiple GPUs**. GPUs within the tray are connected using **high-bandwidth interconnects** such as NVlink. We want, in general, the NVlink to perform better since we use more communication among GPUs.

<figure><img src="/assets/image (107).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/image (108).png" alt="" width="563"><figcaption><p>Processed data are aggregated when all GPUs have finished the computation</p></figcaption></figure>

<figure><img src="/assets/image (110).png" alt="" width="563"><figcaption></figcaption></figure>

**TPUs** (**Tensor Processors Units**) are **ML-specific hardware**, a custom-built integrated circuit developed specifically ML and tailored for TensorFlow. They are used for **training** and **inference**. The basic unit of TPUs are tensors. Each **tensor core** has an **array for matrix computations** (MXU) and a **connection to high bandwidth memory** (HBM) to store **parameters** and **intermediate values** during the computation.

In a rack multiple TPUs v2 accelerator boards are connected through a custom high-bandwidth network, that enables fast parameter reconciliation with well-controlled tail latencies. TPUs v3 is the first liquid-cooled accelerator in Google’s data center. 2.5x faster than TPU v2. TPUs v4 are about 2.7x faster than TPU v3. Same computing capacity as 10 millions of laptops.

<figure><img src="/assets/image (112).png" alt=""><figcaption><p>TPU v2</p></figcaption></figure>

<figure><img src="/assets/image (113).png" alt=""><figcaption><p>TPU v3</p></figcaption></figure>

**FPGAs** (Field-Programmable Gate Array) are **array of logic gates** that can be programmed by the user of the device. Array of carefully designed and interconnected digital sub-circuits that efficiently implement common functions offering very high levels of flexibility. The digital sub-circuits are called **configurable logic blocks** (CLBs).

<figure><img src="/assets/image (115).png" alt="" width="419"><figcaption></figcaption></figure>

#### CPU, GPU, TPU and FPGA: an AI comparison

<figure><img src="/assets/image (116).png" alt="" width="563"><figcaption></figcaption></figure>

## Storage

In 80s-90s data was primarily generated by humans. Nowadays machine generate data at an unprecedented rate (in the order of 100 ZB per year). The growth favors the **centralized storage strategy**: limiting redundant data, automatizing replication and backups, reducing management costs.\
Storage technologies: HDDs, SSDs, NVMe (non-volatile memory express) and tapes.

SSD are not completely replacing HDD, but they are used together.

Some HDD manufacturers produce Solid State Hybrid Disks (SSHD) that combine a small SDD with a large HDD in a single unit. Some large storage servers use SSD as a cache for several HDD. Some main boards of the latest generation have the same feature: they combine a small SSD with a large HDD to have a faster disk.

### Disk abstraction

**Disk** are seen by the OS as a **collection of data blocks** that can be read or written independently. \
In order to allow the management, each block is characterized by a **unique numerical address** called **LBA** (Logical Block Address). Typically, the OS **groups blocks into clusters** to simplify the access to the disk (mapping between where the information is physically located and how the OS access them). Cluster are the **minimal unit that an OS can read from/write to** a disk. Typical cluster size range from 1 disk sector (512 B) to 128 sectors (64 KB).

<figure><img src="/assets/image (117).png" alt="" width="256"><figcaption><p>As we can see, the disk can contain several clusters.</p></figcaption></figure>

Clusters contains two types of information:

* **File data**: the actual content of the files.
* **Meta data**: additional information required to **support** the **management** of the system. It contains file names, directory structures and symbolic links, file size and file type, creation, modification, last access dates, security information (owners, access list, encryption), and links to the LBA where the file content can be located on the disk.

<figure><img src="/assets/image (118).png" alt="" width="563"><figcaption></figcaption></figure>

Clusters are used to simplify/reduce the memory associated with the meta data.  Since the file system can only access clusters, the real occupation of space on a disk for a file is always a multiple of the cluster size. \
Given the file size (s), the cluster size (c) and the actual size on the disk (a) then it follows that $$a = ceil(s/c) * c$$. Then the quantity $$w = a - s$$ is a **wasted disk space** due to the organization of file into clusters. This waste is called **internal fragmentation** of files.

* **Reading** a files requires to: 1) accessing the meta-data to locate its block, 2) access the block to read its content.
* **Writing** a file requires to: 1) accessing the meta-data to locate free-space, 2) write the data in the assigned blocks, 3) update the meta-data.\
  There might **not be enough space** to **store a file contiguously**. In this case, the file is split into smaller chunks that are inserted into the free clusters spread over the disk. The effect of splitting a file into non-contiguous clusters is called **external fragmentation**. This can reduce a lot the performance of an HHD.
* **Deleting** a file requires to only update the meta-data to say that the blocks where the file was stores are no longer in use by the OS. \
  Deleting a file never actually deletes the data on the disk: when a new file will be written on the same clusters, the old data will be replaced by the new one.

### HHD

An **hard drive disk** (HDD) is a data storage using **rotating disks** (**platters**) **coated with magnetic material**. Data is read in a **random-access manner**, meaning individual blocks of data can be stored or retrieved in any order rather than sequentially. An HHD consists of one or more rigid (”hard”) rotating disks (platters) with magnetic heads arranged on a **moving actuator arm** to read and write data to the surfaces.

<figure><img src="/assets/image (121).png" alt="" width="563"><figcaption></figcaption></figure>

It is very fragile. A single pieces of dust can impact how the head read/write the information. If the head crush the data of the HDD will not be recover.

Externally, HDD expose a large number of sectors (blocks) typically 512B or 4096B. Individual sector writes are atomic, they have an header and an error correction code. Multiple sectors writes may be interrupted (**torn write**). **Sectors are arranged into tracks**. A **cylinder** is a particular **track on multiple platters**. Tracks are arranged in concentric circles on platters. A disk may have multiple, double-sided platters.

Many disks incorporate **caches** (**track buffer**), a small amount of RAM (8, 16 or 32 MB). **Read caching** reduces read delays due to seeking and rotation. \
**Write back cache:** drive reports that writes are complete after they have been cached. Possibly dangerous feature. It can create inconsistency. \
**Write through cache:** drive reports that writes are complete after they have been written to disk. \
Today, some disks include **flash memory** for **persistent caching** (hybrid drives). This means that the cache is not flushed when removing the power.

<figure><img src="/assets/image (123).png" alt="" width="375"><figcaption></figcaption></figure>

There could be 4 types of **delay**:

1. **Rotational delay**: time to rotate the desired sector to the read head. Related to the RPM (round per minute). Full rotation delay is $$R = {1\over DiskRPM}$$. In seconds $$R_{sec} = 60* R$$ . The rotational delay is $$T_{rotation_{AVG}} = R_{sec}/2$$. The 2 comes from the fact that in general we have to rotate among half the disk.
2.  **Seek delay**: time to move the read head to a different track. It is characterized by different phases: accelleration, coasting (constant speed), decelleration, settling. $$T_{seek}$$ modelling consider a linear dependency with the distance. $$T_{seek_{AVG}} = T_{seek_{MAX}}/3$$.&#x20;

    <figure><img src="/assets/image (124).png" alt="" width="375"><figcaption></figcaption></figure>
3. **Transfer time**: time to read or write bytes. It is the final phase of the I/O that takes place. It includes the time for the head to pass on the sectors and the I/O transfer.
4. **Controller overhead**: overhead for the request management. Buffer management (data transfer) and interrupt sending time.

The **service time** is $$T_{I/O} = T_{seek} + T_{rotation} + T_{transfer} + T_{overhead}$$.

The **response time** is $$R = T_{I/O} + T_{queue}$$, where $$T_{queue}$$ depends on the queue length, resource utilization, mean and variance of disk service time (distribution) and request arrival distribution.

The **pessimistic case** is the one in which sectors are fragmented on the disk in the worst possible way, thus each access to a sector requires to pay $$T_{seek}$$ and $$T_{rotation}$$. \
In many circumstances this is not the case: files are larger than a block and they are stored in a contiguous way. We can measure the **data locality** of a disk as the percentage of blocks that do not need seek or rotational latency to be found. In this case the average transfer time is $$T_{I/O} = (1 - DL)*(T_{seek} + T_{rotation}) + T_{transfer} + T_{controller}$$.

**Caching** helps improve disk performance, but another limit is poor random access times.

* If there is a queue of requests to the disk, they can be reordered to improve performance.
* Estimation of the request lenght is feasible knowing the position on the disk of the data.
* Several scheduling algorithms can be implemented to improve performance:
  * **FCFC** (first come first serve): most basic one, serve requests in order.\
    The disadvantage is that a lot of time is spent seeking.&#x20;
  * **SSTF** (shortest seek time first): in this way we are **minimizing the total seek time**. \
    The problem is that it is prone to **starvation**: if we receive multiple requests in the same scenario it could happen that we get stuck there, not exploring at all some other requests that are far form the current scenario.\
    The advantage is that it is optimal and it can be easily implemented.
  * **SCAN** (elevator algorithm): head sweeps across the disk servicing requests in order\
    The advantage is that it has reasonable performance and is no prone to starvation. \
    On the other hand the average access time are less for requests at high and low addresses.&#x20;
  * **C-SCAN**: like SCAN, but only service request in one direction (circular SCAN). \
    It is fairer than SCAN but has worst performance.&#x20;
  * **C-LOOK**: C-SCAN variant that peeks at the upcoming addresses in the queue. The head only goes as far as the last request.

### SSD

In a solid-state drive there are **NO mechanical components** (it is a plus since the most delay is associated with the moving parts). It is build out of **transistors** (like memory and processors).

<figure><img src="/assets/image (83).png" alt="" width="375"><figcaption></figcaption></figure>

It retain information despite power loss unlike typical RAM. \
A controller is included in the device with one or more solid state memory components. \
It uses traditional HDD interfaces (protocol and physical connectors) and form factors.&#x20;

In general, it has higher performance than HDD. In theory we expect a 15-20x improvement w.r.t. the HDD. In practice we get the same performance.

Storing bits:

* Single-level cell (SLC)
* Multi-level cell (MLC)
* Triple-level cell (TLC)
* QLC, PLC, ..

<figure><img src="/assets/image (84).png" alt=""><figcaption></figcaption></figure>

Only empty pages can be written. Only dirty pages can be erased, but this must be done at the block level. It is meaningful to read only pages in use. If no empty page exists, some dirty page must be erased. If no block containing just dirty or empty pages exists, then special procedures should be followed to gather empty pages over the disk. To erase the value in flash memory the original voltage must be reset to neutral before a new voltage can be applied, known as write amplification.

Remark: we can write and read a single page of data from a SSD, but we have to delete an entire block to release it.&#x20;

This **mismatch** is one of the cause for the **write amplification** problem: the actual amount of information physically written to the storage media is a multiple of the logical amount intended to be written. Write amplification degrades a lot the performance of an SSD as time passes.

<figure><img src="/assets/image (85).png" alt="" width="375"><figcaption></figcaption></figure>

Another problem is that **flash cells wear out** due to the break down of the oxide layer within the floating-gate transistors of a NAND flash memory. The erasing process hits the flash cell with a relatively large charge of electrical energy. So each time a block is erased:

* The large electrical charge actually degrades the silicon material.
* After enough write-erase cycles, the electrical properties of the flash cell begin to break down and the cell becomes unreliable.

Direct mapping between logical to physical pages is not feasible. **Flash Transition Layer** (**FTL**) is an SSD component that make SSD “look as HDD”. It tries to make a sort of **load balancing**. It manages:

* **Data allocation** and **address translation**:
  * Efficient to reduce write amplification effects.
  * Program pages within an erased block in order (from low to high pages).
*   **Garbage collector**:

    * Reuse of pages with old data (dirty/invalid). \
      Old version of data are called garbage and (sooner or later) they must be reclaimed for new writes to take place. Garbage collection is the process of finding garbage blocks and reclaiming them. \
      It is a simple process for fully garbage blocks, but more complex for partial cases (find a suitable partial block, copy non-garbage pages, erase the entire block for writing).

    <figure><img src="/assets/image (86).png" alt=""><figcaption></figcaption></figure>
* &#x20;**Wear leveling**: FTL should try to spread writes across the blocks of the flash ensuring that all of the blocks of the device wear out at roughly the same time.

Garbage collection is expensive: it requires reading and rewriting of live data, the ideal case is when the entire block consist of only dead pages. The cost depends on the amount of data blocks that have to be **migrated**. Some solutions to alleviate the problem:

* **Overprovision** the device by adding **extra flash capacity** that can be used by the garbage collector (and also for robustness).
* Run the garbage collector in the **background** using less busy periods for the disk. In this case the SSD assumes to know which pages are invalid. \
  Problem: most file systems don’t actually delete data (ex. in Linux the delete function is unlink() and it removes the file meta-data, but not the file itself). For this reason a new command was introduced: **TRIM**. The OS tells the SSD that specific LBAs are invalid and may be gargbage-collected.

Efficient mapping algorithms in the FTL can significantly improve the performance of an SSD. By minimizing the amount of data that needs to be moved around and optimizing the use of available memory, the SSD can achieve faster read and write speeds.

Moreover, the size of a page-level mapping table can be too large. **Reducing the size** of this mapping table **without sacrificing performance** is crucial for making efficient use of memory resources. Some approaches are:

* **Block based mapping**: the idea is to reduce the size of a mapping table by mapping at block granularity. A small problem is that the FTL must read a large amount of live data from the old block and copy them into a new one.
*   **Hybrid mapping**: FTL maintains two tables:

    * **Log block**: page mapped (log → page).
    * **Data blocks**: block-mapped (data → block).

    When looking for a particular logical block, the FTL will consult the page mapping table and block mapping table in order.
* **Page mapping plus caching**: exploiting data locality. The basic idea is to cache the active part of the page-mapped FTL: if a given workload only accesses a small set of pages, the translations of those pages will be stored in the FTL memory. It has high performance without high memory cost id the cache can contain the necessary working set. Cache miss overhead exists.

Erase/Write cycle is limited in flash memory: skewness in the EW cycles shortens the life of the SSD, and all blocks should wear out at roughly the same time. Log-structured approach and garbage collection helps in spreading writes. However, a block may consist of cold data. The FTL must periodically read all the live data out of such blocks and re-write it elsewhere. **Wear leveling** increases the write amplification of the SSD and decreases performance.

In summary:

* SSD cost more than conventional HDD.
* Flash memory can be written only a limited number of times (wear):
  * It have a shorter lifetime.
  * Error correcting codes are necessary.
  * Overprovisioning and some spare capacity.
* Different read/write speed: write amplification problem.
* Write performance degrades of one order of magnitude after the first writing.
* Often the controller become the real bottleneck to the transfer rate.
* SSD are not affected by data-locality and **must not be defragmented** (actually, defragmentation may damage the disks).
* FTL is one of the key components (data allocation, address translation, garbage collection, wear leveling).

#### HDD vs SSD

* **Unrecoverable Bit Error Ratio** (**UBER**): a metric for the rate of occurrence of data errors, equal to the number of data errors per bits read.
* **Endurance rating**: terabytes written (**TBW** is the total amount of data that can be written into an SSD before it is likely to fail). The number of TB that may be written to the SSD while still meeting the requirements.

<div align="center"><figure><img src="/assets/image (87).png" alt="" width="563"><figcaption><p>HDD: the UBER is fixed in time<br>SSD: at the beginning of their lifetime they are very robust, but then they became less robust.<br>This is why we use the SSD for storing information that we do not want to rewrite frequently (such as OS and programs), and we use HHD for data that we will rewrite often. This is also why we use both SSD and HHD at the same time.</p></figcaption></figure></div>

Memory cells can accept data recording between 3’000 and 100’000 during its lifetime. Once the limit value is exceeded, the cell “forgets” any new data. A typical TBW for a 250 GB SSD is between 60 and 150 TB of data written to the drive. It is difficult to comment on the duration of SSDs.

### Storage systems: DAS, NAS and SAN

* A **direct attached storage** (DAS) is a storage system directly attached to a server or workstation. They are visible as disks/volumes by the client OS.
  * Main features: limited scalability, complex management, to read files in other machines (file sharing) protocol of the OS must be used. \
    Internal and external: DAS does not necessarily mean internal drives. All the external disks, connected with a point-to-point protocol to a PC can be considered as DAS.
*   A **network attached storage** (NAS) is a computer connected to a network that provides only **file-based data storage** **services** (ex. FTP, Network File System and SAMBA) to other devices on the network and is visible as File Server to the client OS.

    * NAS systems **contain one or more hard disks**, often organized into logical redundant storage containers or RAID. \
      It provide file-access services to the hosts connected to a TCP/IP network though Network File System and SAMBA. Each NAS element has its own IP address. \
      Main features: good scalability (incrementing the devices in each NAS element or incrementing the number of NAS elements).

    The key difference between DAS and NAS. \
    DAS is simply an extension of an existing server and is not necessarily networked. \
    NAS is designed as an easy and self-contained solution for sharing files over the network. The performance of NAS depends mainly on the speed of and congestion on the network.



    <figure><img src="/assets/image (89).png" alt="" width="563"><figcaption></figcaption></figure>
*   A **storage area networks** (SAN) are remote storage units that are connected to a server using a specific networking technology and are visible as disks/volumes by the client OS.

    * SANs have a special network devoted to the accesses to storage devices: two distinct networks one TCP/IP and one dedicated network (ex. Fiber Channel). \
      Main feature: high scalability (simply increasing the storage devices connected to the SAN network).

    The key difference between NAS and SAN. \
    NAS provides both storage and file storage. They appear to the client OS as a file server (the client can map network drives to shares on that server). Traditionally used for low-volume access to a large amount of storage by many users. \
    SAN provides only **block-based storage** and leaves file storing concerns on the “client” side. A disk available through a SAN still appears to the client OS as a disk: it will be visible in the disks and volumes management utilities (along with client's local disks), and available to be formatted with a file system. Traditionally used for petabytes of storage and multiple, simultaneous access to files, such as streaming audio/video.

<figure><img src="/assets/image (90).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/image (91).png" alt="" width="563"><figcaption></figcaption></figure>

### RAID (redundant arrays of independent disks)

**Redundant arrays of independent** (inexpensive) **disks** (RAID) were proposed for the need to increase the performance, the size and the reliability of storage systems. They are several independent disks that are considered as a single, large, high-performance logical disk.\
The data are **striped across several disks accessed in parallel**:

* **High data transfer rate**: large data accesses (heavy I/O op.)
* **High I/O rate**: small but frequent data accesses (light I/O op.)
* **Load balancing** across the disks.

Two orthogonal techniques:

1.  **Data striping**: to improve **performance**. It is not very robust: if only one disk fail, the entire information is lost. **I/O virtualization**: data are distributed transparently over the disks (no action is required to the users by the OS).&#x20;

    <figure><img src="/assets/image (69).png" alt="" width="375"><figcaption><p>2 Byte interleaving (stripe unit)</p></figcaption></figure>
2. **Redundancy**: to improve **reliability**. It is necessary to overcome the problem of data striping.

In data striping data are written sequentially in units (stripe unit: bit, byte, blocks, ..) on multiple disks according to a cyclic algorithm (**round robin**).\
The strip width is the number of disks considered by the striping algorithm:

* **Multiple independent I/O requests** will be executed in **parallel** by **several disks** decreasing the queue length (and time) of the disks.
* **Single multiple-block I/O requests** will be executed by **multiple disks** in **parallel** increasing of the transfer rate of a single request.

The more physical disks in the array, the larger the size and performance gains but the larger the probability of failure of a disk. This is the main motivation of the introduction of **redundancy**: **error correcting codes** (stored on disks different from the ones with the data) are computed to **tolerate loss** due to disk **failures**. Since write operations must update also the redundant information, their performance is worse than the one of the traditional writes.

Hard drives are great devices (relatively fast, persistent storage) but we want to cope with disk failures (mechanical parts break over time, sectors may become silently corrupted), and with the limited capacity (managing files across multiple physical devices is cumbersome).

Instead RAID use multiple disks to create the illusion of a large, faster, more reliable disk. Externally it looks like a single disk. Data blocks are read/written as usual. No need for software to explicitly manage multiple disks or perform error checking/recovery. Internally it is a **complex computing system**: disks are managed by a dedicated CPU + software, it has RAM and non-volatile memory, many different configuration options (RAID levels).

<figure><img src="/assets/image (70).png" alt="" width="563"><figcaption><p>Standard RAID levels</p></figcaption></figure>

#### RAID level 0: striping, no redundancy

Data are written on a **single logical disk** and **splitted** in **several blocks** distributed across the disks according to a striping algorithm. It is used where performance and capacity, rather than reliability, are the primary concerns.\
Key idea: present an array of disks as a single large disk. Maximize parallelism by striping data cross all N disks.

(+) **Lowest cost** because it does not employ redundancy (no error-correcting codes are computed and stored).\
(+) **Best write performance** (it does not need to update redundant data and it is parallelized).

(-) **Single disk failure will result in the data loss**.&#x20;

<figure><img src="/assets/image (71).png" alt="" width="563"><figcaption></figcaption></figure>

The disk number is equal to the logical block number. The offset is equal to the logical block number divided by the number of disks.

Chuck size impact array performance: smaller chunks -> greater parallelism, big chunks -> reduced seek time.

#### Raid level 1: mirroring

Whenever data is written to a disk it is also **duplicated** (mirrored) to all other disks (there are always **multiple copies** of the data). At minimum 2 disk drives are needed.

(+) **High reliability**: when a disk fails the second copy is used.\
(+) **Fast read**: it can be retrieved from the disk with the shorter queueing, seek, and latency delays.\
(+) **Fast writes** (no error correcting code should be computed), but still slower than standard disks (due to duplication).

(-) **High costs** (only 50% of the capacity is used).&#x20;

Theoretically, a RAID 1 can mirror the content over more than one disk:

* It gives resiliency to errors even if more than one disk breaks.
* It allows with a voting mechanism to identify errors not reported by the disk controller.

In practice this is never used, because the **overhead** and **costs** are too **high**. RAID 0 offers high performance, but zero error recovery. Key idea: make two copies of all data. However, if several disks are available (even number), disks could be coupled. The total capacity is halved. Each disk has a mirror.&#x20;

RAID levels can be **combined**.

<figure><img src="/assets/image (72).png" alt="" width="563"><figcaption></figcaption></figure>

* RAID level 0 + 1: group of **striped** disks that are **then mirrored**. A minimum of 4 drives is needed. After the first failure the model becomes a RAID 0.
* RAID level 1 + 0: group of **mirrored** disks that are **then striped**. A minimum of 4 drives is needed. It is used in DB with very high workloads (fast writes).

<div><figure><img src="/assets/image (73).png" alt="" width="188"><figcaption></figcaption></figure> <figure><img src="/assets/image (74).png" alt="" width="188"><figcaption></figcaption></figure></div>

The blocks are the same but are allocated in a different order. **Performance** and **storage capacity** on both RAID 10 and RAID 01 are the **same**. The main difference is the **fault tolerance level**: in RAID 01 fault tolerance is less than in RAID 10.

Mirrored writes should be **atomic** (all or nothing), but this is diffult to guarantee. Many RAID controller include a **write-ahead log** that consist of a battery backed, non-volatile storage of pending writes and a recovery procedure ensures to recover the out-of-sync mirrored copies.

The point is that RAID 1 offers highly reliable data storage, but it uses N/2 of the array capacity. We can achieve the same level of reliability without wasting so much capacity by using information coding techniques to build **light-weight error recovery mechanisms**.

#### RAID level 4: parity drive

A drive is used to store parity information: additional data calculated from the original data to provide error detection and correction capabilities. It is used to reconstruct data in case of a drive failure.

Reads (serial or random) are not a problem in RAID 4. Random writes in RAID 4:

1. Read the **target** block and the **parity block**.
2. Use subtraction to **calculate** the **new parity** block.
3. Write the target block and the parity block.

RAID 4 has a terribile write performance: **bottle-necked** by the **parity drive**. On the contrary, RAID 4 provides good performance of random reads thanks to parallelization across all non-parity blocks in the stripe (allow multiple simultaneous reading operations).

<figure><img src="/assets/image (75).png" alt="" width="563"><figcaption></figcaption></figure>

#### RAID level 5: rotating parity

Random writes in RAID 5:

1. Read the target block and the parity block.
2. Use subtraction to calculate the new parity block.
3. Write the target block and the parity block.

So a total of 4 operations (2 reads, 2 writes) distributed across all drives.

<figure><img src="/assets/image (77).png" alt="" width="563"><figcaption></figcaption></figure>

#### **RAID level 6:**&#x20;

More fault tolerance w.r.t. RAID 5:

* 2 concurrent failures are tolerated.
* It uses Solomon-Reeds codes with two redundancy schemes (P+Q) distributed and independent.
* N+2 disks are required (since there are 2 parity blocks per stripe)
* It has high overhead per writes, since each write require 6 disk accesses due to the need to update both the P and Q parity blocks (slow writes).
* Minimum set of 4 data disks.

<figure><img src="/assets/image (76).png" alt="" width="563"><figcaption></figcaption></figure>

Many RAID systems include a hot spare (idle, unused disk installed in the system), if a drive fails the array is immediately rebuilt using the hot spare.

RAID can be implemented in hardware or software: hardware is faster and more reliable but migration (to another hardware RAID) almost never work, on the contrary software arrays are simpler to migrate and cheaper, but have worse performance and weaker reliability.

## Networking

#### Stages of enterprise infrastructures

1. Monolithic app: minimal network demands, proprietary protocol.
2. Client-Server: high network demand inside the enterprise, applications walled within the enterprise, TCP/IP + proprietary protocol.
3. Web applications: pervasive TCP/IP, access from anywhere, servers are broken into multiple units.
4. Microservices: infrastructures moved to cloud providers, servers broken into microservices, increase of server-to-server traffic.

### Datacenters aka warehouse scale computing (WSC)

A **datacenter** is the set of all the **physical infrastructure** required to **support** a **cloud computing service**. The whole infrastructure is **co-located** either in a room, or in a building, or in a set of adjacent building. \
Applications: cloud computing, cloud storage, web services. It is based on the consolidation of computation and network resources.

The performance of servers increases over time, but the **demand for inter-server bandwidth** increases as well. We can double the aggregate compute capacity or the aggregate storage simply by doubling the number of computer or storage elements.

Networking has **no straightforward horizontal scaling solution.** **Doubling leaf bandwidth** is easy: with **twice as many servers**, we will have twice ad many network ports and thus twice as much bandwidth. But if we assume that **every server needs to talk to every other server**, we need to deal with **bisection bandwidth**. Bisection bandwidth is the bandwidth across the narrowest line that equally divides the cluster into two parts. It characterizes network capacity since randomly communication processors must send data across the “middle” of the network.

DCNs can be classified into three main categories:

1. **Switch-centric** architectures: uses switches to perform packet forwarding.
2. **Server-centric** architectures: uses servers with multiple Network Interface Cards (NICs) to act as switches in addition to performing other computational functions.
3. **Hybrid** architectures: combine switches and servers for packet forwarding.

### Switch-centric architectures

<figure><img src="/assets/image (42).png" alt="" width="548"><figcaption></figcaption></figure>

Within a DCN traffic can be divided into North-South traffic and East-West traffic. The first is the traffic from the outside the DCN to a server within.

East-west traffic consist of storage replication, VM migration, network function virtualization (NFV). It can be unicast (point-to-point, ex. data backup, VM migration), multicast (many-to-many ex. software update, data replication, OS image provisioning) or incast (many-to-one, ex. merging tables in DB).\
East-west traffic is usually larger than north-south traffic.

#### Three-tier "classical" network

<div><figure><img src="/assets/image (43).png" alt=""><figcaption></figcaption></figure> <figure><img src="/assets/image (49).png" alt=""><figcaption></figcaption></figure></div>

It is the simplest and most widely used topology.\
Servers are connected to the DCN through access switches. Each access-level switch is connected to at least two aggregation-level switches. Aggregation-level switches are connected to core-level switches (gateways). \
It reflects the topology of the datacenter (physical layout).

Bandwidth can be increased by increasing the switches at the core and aggregation layers, and by using rotating protocols such as equal cost multiple path (ECMP) that equally shares the traffic among different routes.

It can be very expensive in large data-centers since upper layers require faster network equipment, each layer is implemented by switches of a different kind so the cost in term of acquisition, management and energy consumption can be very high.&#x20;

#### ToR (top of rack) architecture

In a rack, all servers are connected to a ToR (top of rack) access switch. The servers and the ToR switch are co-located in the same rack while aggregation switches are in dedicated racks or in shared racks with other ToR switches and servers. ToR uses over-subscription.\
Limited scalability, higher complexity for switch management (high number of switches) but simpler cable management.

<figure><img src="/assets/image (50).png" alt=""><figcaption></figcaption></figure>

#### EoR (end of row) architectures

Aggregation switches are positioned one per corridor, at the end of a line (row). Servers in a racks are connected directly to the aggregation switch in another rack. Aggregation switches must have a larger number of ports. \
More complex cabling but simpler switch management and scalable.&#x20;

<figure><img src="/assets/image (51).png" alt=""><figcaption></figcaption></figure>

#### Leaf-spine architecture&#x20;

It is made of two stage interconnections: ToR switches (leaf) and dedicated (aggregation) switches (spine).\
Each stage is fully interconnected. It is a non-folded clos structure.&#x20;

<figure><img src="/assets/image (44).png" alt=""><figcaption></figcaption></figure>

The basic idea is that we have an input stage, a middle stage and an output stage, the input is connected to the output via the middle stage in such a way that it is always possible to have a path from one input to an output.

The main advantage is the use of homogeneous equipment, that there is no need for forwarding due to existence of direct paths (ECMP strategy with a rounding protocol is used), and that the number of hops is the same for any pair of nodes.

Clos network can be scaled to multi-tier network. An option is to transform each spine-leaf group into a pod and add a super-spine tier. It is an highly scalable and cost-efficient DCN architecture that aims to maximize bisection bandwidth.

Let k be the number of middle stages switches, and n be the number of input and outputs of switches of side stages.&#x20;

* if $$k \ge n$$ there is always a way to rearrange communications to free a path between any pair of idle input/output.
* if $$k \ge 2n - 1$$ there is always a free path between any pair of idle  input/output.&#x20;

NB: t is a free design parameter, the total number of input/output $$N = t \cdot n$$ can scale freely by increasing the size of middle-stage switches.

<figure><img src="/assets/image (45).png" alt=""><figcaption></figcaption></figure>

#### Clos topology (n = m = k)

Each switching module is unidirectional: k input + k output ports per module.&#x20;

#### Leaf and spine topology

Each switching module is bidirectional.\
Each leaf has t switching modules with 2k bidirectional ports per module.\
Each spine has k switching modules with t bidirectional ports per module.

<div><figure><img src="/assets/image (46).png" alt=""><figcaption><p>Clos topology</p></figcaption></figure> <figure><img src="/assets/image (47).png" alt=""><figcaption><p>Leaf and spine topology</p></figcaption></figure></div>

#### PoD (point of delivery)

A PoD is a module or group of network, compute, storage, and application components that work together to deliver a network service. Each PoD is a repeatable pattern, and its components increase the modularity, scalability, and manageability of data.&#x20;

It is a leaf with $$2k^2$$ bidirectional ports: $$k^2$$ ports to the server and $$k^2$$ports to the datacenter network.&#x20;

<figure><img src="/assets/image (48).png" alt="" width="209"><figcaption></figcaption></figure>

The PoD datacenter network architecture is highly scalable and cost-efficient, utilizing a pod-based model known as the Fat Tree. This model aims to maximize bisection bandwidth.

#### Fat-tree network

At the edge layer, there are $$2k$$ PoDs (groups of servers) each with $$k^2$$ severs.\
Each edge switch is directly connected to k servers in a pod and k aggregation switches. This enable partial connectivity at switch level allowing for efficient intra-pod and inter-pod communications.

<figure><img src="/assets/image (52).png" alt=""><figcaption></figcaption></figure>

#### VL2 network

It is a cost-effective hierarchical fat-tree-based DCN architecture with high bisection bandwidth. It uses 3 types of switches: intermediate, aggregation, and ToR switches.&#x20;

It uses $$D_A/2$$ intermediate switches, $$D_I$$ aggregation switches and $$D_A \cdot D_I / 2$$ ToR switches. The number of servers is $$20 (D_A \cdot D_I)/4$$.\
It uses a load-balancing technique called valiant load balancing (VLB).

<figure><img src="/assets/image (56).png" alt=""><figcaption></figcaption></figure>

### Server centric architecture

It may reduce implementation and maintenance costs by using only servers to build the DCN. \
It uses a 3D-Torus topology to interconnect the server directly. In this way, it exploits network locality to increase communication efficiency.&#x20;

The drawback is that servers with multiple NICs are required to assemble a 3D-Torus network, leading to long paths and high routing complexity.&#x20;

<figure><img src="/assets/image (54).png" alt="" width="311"><figcaption></figcaption></figure>

### Hybrid architecture

<figure><img src="/assets/image (55).png" alt=""><figcaption></figcaption></figure>

## Building level

In general, we expect the datacenter to be located all in the same place, but this is not always the case.\
WSC has other important components related to power delivery, cooling, and building infrastructure that also need to be considered.

<figure><img src="/assets/image (78).png" alt="" width="563"><figcaption></figcaption></figure>

### Cooling systems

IT equipment generates a lot of heat: the cooling system is usually a **very expensive** component of the datacenter, and it is composed by **coolers**, **heat exchangers** and **cold water tanks**.

The simplest topology is fresh air cooling (essentially opening the windows). This is a **single open loop** system also called **free cooling**, as it refers to the use of cold outside air to either help the production of chilled water or directly cool servers. It is not completely free (zero cost), but it involves very low-energy costs compared to chillers.

**Closed loop** systems come in many forms, the most common being the air circuit on the data center floor. The goal is to isolate and remove heat from the servers and transport it to a heat exchanger. Cold air flows to the servers, heats up, and eventually reaches a heat exchanger to cool it down again for the next cycle through the servers.

In closed loop systems with **two loops**:

* The airflow through the underfloor plenum, the racks, and back to the CRAC (computer room air conditioning) defines the primary air circuit, that is the first loop.
* The second loop (the liquid supply inside the CRACs units) leads directly from the CRAC to external heat exchangers (typically placed on the building roof) that discharge the heat to the environment.

<figure><img src="/assets/image (57).png" alt="" width="563"><figcaption><p>the second loop is inside the CRAC</p></figcaption></figure>

A three loop system is commonly used in large-scale data centers.

<figure><img src="/assets/image (79).png" alt="" width="563"><figcaption></figcaption></figure>

Each topology presents trade-offs in complexity, efficiency, and cost:

* Fresh air cooling can be very **efficient** but d**oes not work in all climates**, **requires filtering** of airborne particulates, and can introduce complex control problems.
* Two-loop systems are **easy to implement**, relatively inexpensive to construct, and offer isolation from external contamination, but typically have **lower operational efficiency**.
* A three-loop system is the **most expensive** to construct and has moderately complex controls, but **offers contaminant protection** and **good efficiency**.

**In-rack cooler** adds an air-to-water heat exchanger at the back of a rack so the hot air exiting the servers immediately flows over coils cooled by water, essentially reducing the path between server exhaust and CRAC input.\
**In-row cooling** works like in-rack cooling except the cooling coils are not in the rack, but adjacent to the rack.

We can directly cool servers components using **cold plates** (local liquid-cooled heat sinks) but, in general, they are impractical.

The liquid circulating through the heat sinks transports the heat to a liquid-to-air or liquid-to-liquid heat exchanger that can be placed close to the tray or rack, or be part of the data center building (such as a cooling tower).

**Container-based datacenters** go one step beyond in-row cooling by placing the server racks inside a physical container (typically 6-12m long) and integrating heat exchange and power distribution into the container as well.

<figure><img src="/assets/image (80).png" alt="" width="375"><figcaption></figcaption></figure>

### Power supply

In order to protect against power failure, **battery** and **diesel generators** are used to **backup** the **external supply**. The batteries are designed in order to guarantee 30s of autonomy, that are needed for the generator to start.

The UPS (uninterruptible power suppl&#x79;_)_ typically combines three functions in one system:

* Contains a **transfer switch** that **chooses** the **active power input** (either utility power or generator power).
* Contains some form of **energy storage** (electrical, chemical, mechanical) to bridge the time between the utility failure and the availability of generator power.
* Apply some conditions on the incoming power, removing voltage spikes or sags, or harmonic distortions in the AC feed.

Datacenters power consumption is an issue, since it can reach several MWs. Cooling usually requires about half the energy required by the IT equipment (server + network + disks). Energy transformation creates also a large amount of energy wasted for running a datacenter. DCs consume 3% of global electricity supply. DCs produce 2% of total greenhouse gas emissions.

<figure><img src="/assets/image (81).png" alt="" width="563"><figcaption></figcaption></figure>

**Power usage effectiveness** (**PUE**) is the ratio of the total amount of energy used by a DC facility to the energy delivered to the computing equipment $$PUE = {Total\ Facility\ Power}/{IT\ Equipment\ Power}$$.

Where the total facility power = covers IT systems (servers, network, storage) + other equipment (cooling, UPS, switch gear, generators, lights, fans, etc.).

The **datacenter infrastructure efficiency** (DCiE) = PUE inverse.

### Failure recovery

Data center availability is defined in four different tier level. Each one has its own requirements:

<figure><img src="/assets/image (82).png" alt=""><figcaption></figcaption></figure>

## SW infrastructures

## Cloud computing

Cloud computing is a coherent, **large-scale**, **publicly accessibile** collection of **computing storage**, and **networking resources**. It is **available via web service calls** through the internet. Short or long term access on a pay-per-use basis.

Cloud provisioning helps in reducing the costs and improving performance. Cloud computing is implemented through virtualization.

## Virtualization

**Hardware resources** (CPU, RAM, ecc) are **partitioned** and **shared** among multiple virtual machines (VMs). \
The virtual machine monitor (VMM) governs the access to the physical resources among running VMs.

<figure><img src="/assets/image (128).png" alt="" width="563"><figcaption></figcaption></figure>

A machine is an execution environment capable of running a program. The difference between a physical machine and a VM is in the computer architecture:

* **Sets of instructions** characterized the levels at which are considered. We usually program the software part, exploiting the hardware. The **ISA** (**instruction set architecture**) corresponds to level 2 in the layered execution model. ISA marks the division between hardware and software.
  * **User ISA**: aspects of the ISA that are visible to an application program. When application interacts with the HW, user ISA is used. It collects all instruction that are directly **executed** by an **application program**.
  * **System ISA**: aspects visibile to supervisor software (the OS) which is responsible for managing hardware resources (hide the complexity of CPUs, define how app access memory, communicate with the HW). When the OS interacts with the HW (drivers, MM, scheduling) system ISA is used. It collects all instruction that are **executed** only by the **OS**.

The **ABI** (**application binary interface**) corresponds to level 3 in the layered architecture:

* **User ISA**: aspects of the ISA that are visible to an application program.
* **System calls**: calls that allow programs to interact with shared hardware resources indirectly by OS.

One machine can only run instruction that were meant for it. OS can create new instructions for programs to access devices/hw. An application rely on system call provided by the OS.

<figure><img src="/assets/image (129).png" alt="" width="563"><figcaption></figcaption></figure>

A VM is a **logical abstraction** able to provide a **virtualized execution environment**. More specifically:

* Provides **identical software behavior**.
* Consist in a combination of **physical machine** and **virtualizing software**.
* May appear as **different resources** than physical machine.
* May result in **different level of performance**.

Its tasks are:

* To **map virtual resources** or states to corresponding physical ones.
* To **use physical machine instruction/calls to execute the virtual ones**.

There are two types of VM.

### System VM

The VMM supports the **level 0-2** of the previous architecture.

* The virtualization software is caller **VMM** (**virtual machine monitor**). The VMM can provide its functionality either working directly on the hardware, or running another OS.
* It provide a **complete system environment** that can **support an OS**, potentially with many user processes.
* It **provides OS** running in it access to underlying hardware resources (networking, I/O, a GUI).
* The VM supports the OS **as long as the system environment is alive**.
* The **virtualizing software** is placed between hardware and software and it **emulates** the **ISA interface** **seen by software**.

### Process VM

The runtime software supports the **level 0-3** of the architecture.

* The virtualization software is usually called **runtime software**.
* It is able to support an **individual process**.
* It is placed at the **ABI interface**, on top of the OS/hardware combination.
* It **emulates both user-level instructions** and **operating system calls**.

<figure><img src="/assets/image (131).png" alt="" width="375"><figcaption></figcaption></figure>

#### Differences between system/process VM

The differences relies on the difference between **host** (the underlying platform supporting the environment/system) and **guest** (the software that runs in the VM environment as the guest).

<figure><img src="/assets/image (133).png" alt="" width="563"><figcaption></figcaption></figure>

<figure><img src="/assets/image (135).png" alt="" width="563"><figcaption></figcaption></figure>

#### **Multiprogrammed systems**

Same ABI/OS:

* VM formed by OS call interface + user ISA.
* Common approach of all modern OS for **multi user support** (task/process manager).
* Each user process is given the **illusion of having a complete machine to itself**, its own **address space** and is given **access to a file structure**.
* OS timeshares and manages HW resources to permit this.

#### **HLL** (**high level language**) **VMs**

The goal is to have an **isolated execution environment for each application** or for **different instances** of the same application (ex. Java VM). VM tasks:

* Translates application byte code to OS-specific executable.
* Minimize HW/OS-specific feature for platform independence.

Applications runs normally but are **sandboxed**, can “**migrate**”, are **not conflicting** one another, are **not** “**installed**” in a strict sense.

#### **Emulation**

It refers to those software technologies developed to allow an application (or OS) to run in an environment different from that originally intended. \
It is required when the **VMs** have a **different ISA/ABI from** the architecture where they are **running on** (ex. obsolete hardware emulators, other architecture emulators).

#### **Classic system VM for same ISA**

The **VMM** is on **bare hardware**, and VMs fit on top. The VMM can **intercept guest OS’s interaction with hardware resources**. It is the most **efficient** VM architecture since HW executes the same instructions of VMs. Two different OSs on the same HW.

<figure><img src="/assets/image (136).png" alt="" width="375"><figcaption></figcaption></figure>

#### **Hosted VM**

The virtualizing software is on top of an existing host OS.

<figure><img src="/assets/image (139).png" alt="" width="375"><figcaption></figcaption></figure>

#### **Whole-system VMs**

All software is virtualized: ISAs are different so both application and OS code require emulations (no naive execution possibile). Usually the VMM and guest software is on top of a host OS running on the hardware. The VM software must emulate the entire hardware environment and all the guest ISA operations to equivalent OS call to the host.

<figure><img src="/assets/image (138).png" alt="" width="375"><figcaption></figcaption></figure>

### Virtualization mechanism

The virtualization is implemented by **adding layers between execution stack layers**. Depending on where the new layer is placed, we obtain different types of virtualization.

* **Hardware-level virtualization**: Virtualization layer is placed between hardware and OS. The interface seen by OS and application might be different form the physical one.
* **Application-level virtualization**: A virtualization layer is placed between the OS and some applications (ex. Java VM). It provides the same interfaces to the applications. Applications run in their environment, independently from OS.
* **System-level virtualization**: The virtualization layer provides the interface of a physical machine to a secondary OS and a set of application running in it, allowing them to run on top of an existing OS. It is placed between the system’s OS and other OS. It enables different OSs to run on a single HW.

<div><figure><img src="/assets/image (66).png" alt=""><figcaption><p>HW-level</p></figcaption></figure> <figure><img src="/assets/image (67).png" alt=""><figcaption><p>application-level</p></figcaption></figure> <figure><img src="/assets/image (68).png" alt=""><figcaption><p>system-level</p></figcaption></figure></div>

Properties of virtualization technologies:

<figure><img src="/assets/image (143).png" alt="" width="563"><figcaption></figcaption></figure>

### VMM (virtual machine manger)

It is an application that:

* **Manages** the **VMs**.
* **Mediates** **access** to the **HW** resources on the physical host system.
* **Intercepts** and **handles** any **privileged** or **protected instructions** issued by the virtual machines.

This type of virtualization typically runs VM whose OS, libraries, and utilities have been compiled for the same type of processors and instruction set as the physical machine on which the virtual systems are running.\
&#x20;It provides a user friendly interface to the underlying virtualization software.

Three terms are used to identify the same thing:

* **Virtual machine manager.**
* **Virtual machine monitor.**
*   **Hypervisor**: a virtualization software that runs directly on the hardware. It can be of two types:

    * **Type 1** or **bare metal**: It takes direct control of the hardware. It can be used to provide hacks to some OSs.
    * **Type 2** or **hosted**: It reside within a host OS. It is characterized by at least **two OSs running on the same hardware**. The host OS controls the hardware of the system where the VMM runs. The guest OS is the one that runs in the VM. \
      The **VMM runs in the host OS**, while **applications run in the guest OS** (this type is used when the user wants to interact with the machine). \
      It is **more flexible** in terms of underlying hardware. It is **simpler** to **manage** and **configure** (VMM can use the host OS to provide GUI, not only BIOS). Special care must be taken to **avoid conflict** between host OS and guest OS (ex. virtual memory). The host OS might consume a non negligible set of physical resources (it must be taken into account that resources are shared with the host OS). The guest OS is actually an application running code in a privileged way.

    <figure><img src="/assets/image (145).png" alt="" width="375"><figcaption></figcaption></figure>

The overall architecture can be:

* **Monolithic**: device drivers run within the hypervisor. It provides **better** **performance** and better **isolation**, but can run only on hardware for which the hypervisor has drivers.
* **Microkernel**: device drivers run within a **service VM**. It has smaller hypervisor, it leverages drive ecosystem of an existing OS and it can use 3rd party driver (even if not always easy, recompiling might be required).

<figure><img src="/assets/image (146).png" alt="" width="563"><figcaption></figcaption></figure>

### Virtualization techniques

The way in which the guest OS acts as the host OS. There are two different ways to implement (system level/same ISA) virtualization:

* **Para virtualization**: It is the simpler way to deal with the problem. \
  Idea: modify all the system calls of the guest OS, such that they are not systems calls but regular calls of the host OS. \
  VMM presents to VMs an **interface similar** (but not identical) to that of the **underlying hardware**. In this way, we are modifying how the guest OS interact with the hardware. The goal is to reduce guest’s executions of tasks, that is too expensive for the virtualized environment (we create ”hooks” that allow the guests and host to collaborate with request and acknowledge tasks which would otherwise be executed in the virtual domain, where the execution performance is worse). We modify the structure of the guest OS such that it does not give precedence to the privileged instruction. \
  Advantages: simpler VMM and high performance.\
  Disadvantages: modified guest OS (so it cannot be used with traditional OSs.)
* **Full virtualization**: It provides a **complete simulation** of the **underlying hardware** (full instruction set, I/O operations, interrupts, memory access). Some (privileged) protected instructions are trapped and handled by the hypervisor (VMM). The hypervisor is in charge of managing the virtualization layer. \
  Advantages: unmodified OS. \
  Disadvantages: hypervisor mediation (to allow the guests and host to request and acknowledge tasks which would otherwise be executed in the virtual domain), it does not work on every architecture since it require specific hardware.

They are different in assumptions, but the goal is almost the same. In para-virtualization we do not have access to system-calls: they are modified into hyper-calls that activate the hypervisor (not the hardware directly).

<figure><img src="/assets/image (149).png" alt=""><figcaption></figcaption></figure>

### Containers

Containers (ex. docker) are **pre-configured packages**, with everything you need to execute the code (code, libraries, variables, configurations) in the target machine. \
The main advantage is that their **behavior** is **predictable**, **repeatable** and **immutable**. When I create a “master” container and duplicate it on another server, I know exactly hot it will be executed. There are **no unexpected errors** when moving it to a new machine or between environments.

VM provides hardware virtualization, while containers provide virtualization at the **operating system level**. The main difference is that the containers share the host system kernel with other containers.

<figure><img src="/assets/image (150).png" alt="" width="563"><figcaption></figcaption></figure>

Characteristics:

* **Flexible**: even the most complex applications can be containerized.
* **Light**: the containers exploit and share the host kernel.
* **Interchangeable**: recurrent updates can be distributed on the fly.
* **Portable**: you can create locally, deploy in the cloud and run everywhere.
* **Scalable**: it is possible to automatically increase and distribute replicas of the container.
* **Stackable**: containers can be stacked vertically and on the fly.

Containers **ease the deployment of applications** and **increase the scalability** but also **impose a modular application development** where the modules are independent and uncoupled.

Containers are typically used for:

* Helping make your local development and build workflow faster, more efficient, and more lightweight.
* Running stand-alone services and applications consistently across multiple environments.
* Create isolated instances to run tests.
* Building and testing complex applications and architecture on a local host prior to deployment into a production environment.
* Providing lightweight stand-alone sandbox environments for developing, testing, and teaching technologies, such as the Unix shell or a programming language.
* Software as a Service applications.

<figure><img src="/assets/image (151).png" alt="" width="563"><figcaption></figcaption></figure>

Containers will not replace VM: if we need a different OS then we must use a VM.

## Computing architectures

Virtualization is the basis of data centers. Virtualization provide more to the customers.

* **Without virtualization**: **software is strongly linked with hardware**, so move/change an application is not an easy task. In order to isolate failure/crash, the classical implemented model is: 1 sever, 1 OS, 1 application, with a resulting low CPU utilization (10-15%). Low flexibility.
* **With virtualization**: **SW/HW independence**. **High flexibility** thanks to pre-built VMs. OS and application can be handled as a “single entity”.

**Server consolidation** is the process of migrating network services and applications from physical to virtual machine. It is made by putting an **intermediate layer** of **processing**, that is virtualization, between the resources and the customers.

<figure><img src="/assets/image (152).png" alt="" width="375"><figcaption></figcaption></figure>

It is possible to **automatically balances** the **workloads** according to set limits and guarantees. Server and applications are protected against component and system failure.

Advantages of consolidation:

* Different OS can run on the same hardware.
* Higher hardware utilization: less HW is needed, green IT-oriented, acquiring costs, management costs.
* Continue to use legacy software.
* Application independent from the hardware.

### Cloud computing

Cloud computing is a **model** for enabling **convenient** and **on-demand network access** to a **shared pool** of **configurable computing resources** (networks, servers, storage, applications, services) that can be rapidly provisioned and released with minimal management effort or service provide interaction.

<figure><img src="/assets/image (153).png" alt="" width="375"><figcaption></figcaption></figure>

* **Cloud Application layer** (SaaS): Users access the services provided by this layer trough **web-portals**, and may be required to pay fees to use them. Cloud applications can be developed on the cloud software environments or infrastructure components.
* **Cloud Software Environment layer** (PaaS): Users are application developers. Providers **supply developers** with a **programming-language-level environment with well-defined API**. This facilitates interaction between environment and apps, accelerate the deployment and support scalability.
* **Cloud Infrastructure layer** (IaaS computational, DaaS storage, CaaS communications): It provides resources to the higher-level layers.
  * **Computational** resources (IaaS): they can be **VM**. They are flexible and have **super-user root** for **fine granularity settings** and **customization** of **installed sw** but, on the other hand, they may cause performance interference and cannot provide strong guarantees about SLAs (service level agreements).
  * **Storage** (Data-aaS): It allows users to store their data at remote disks and access it anytime from any place. It facilitates cloud applications to scale beyond their limited server requirements (high dependability, replication, data consistency).
  * **Communication** (CaaS): It is part of a larger category of services known as SaaS, in which vendors offer software products and services over the internet. The core concept of CaaS is that accessing these services over the internet is extremely convenient.

<figure><img src="/assets/image (64).png" alt="" width="431"><figcaption><p>A variety of "as-a-service" offered in Clouds</p></figcaption></figure>

Types of clouds:

* **Private**: used for **single organization**, can be internally or externally hosted. \
  They are **internally managed** data centers. The organization sets up a virtualization environment on its own servers (in its data center, or in the data center of a managed service provider). \
  The advantage is that you have **total control** over every aspect of the infrastructure and you gain the advantages of virtualization. \
  The disadvantages is that is lacks the freedom from capital investment and flexibility. It is useful for companies that have significant existing IT investments.
* **Community**: **shared** by **several** organization, typically externally hosted, but may be internally hosted by one of the organizations. \
  It is a single cloud managed by several federated organizations: this allows economy of scale. Resources can be shared and used by one organization, while the others are not using them. \
  They are technically similar to private cloud since they share the same issues, but a **more complex accounting system** is required.&#x20;
* **Public**: provisioned for open use for the public by a particular organization who also hosts the service. \
  They are **large scale infrastructures** available on a rental basis. OS virtualization provides CPU isolation. Request are accepted and resources granted via web servers, in fact customers access resources remotely via the internet.
* **Hybrid**: composition of two or more clouds that remain unique entities but are bounded together, offering the benefits of multiple deployment models. \
  It can be internally or externally hosted. \
  Usually are companies that holds their private cloud, but that they can be subject to unpredictable peaks of load. The company rents resources from other types of cloud. In order to simplify the development process, the way in which VMs are started, terminated, address is given storage is accessed, must be similar as possible. Many standards are being developed but none is globally accepted yet. Currently, the Amazon EC2 model is the one with more compliant infrastructures.

<figure><img src="/assets/image (62).png" alt=""><figcaption></figcaption></figure>

Advantages of cloud computing:

* Lower IT costs.
* Improved performance.
* Instant software updates.
* “Unlimited” software capacity.
* Increased data reliability.
* Universal data access.
* Device independence.

Disadvantages of cloud computing:

* Requires a constant internet connection.
* Does not work well with low-speed connections.
* Features might be limited.
* Can be slow.
* Stored data might not be secure.
* Stored data can be lost.

### Edge/Fog computing

Edge computing brings computation and data storage closer to the data source, typically at the edge of the network (ex. IoT devices, gateways, local servers, ..). \
Fog computing extends this concept by creating a distributed computing infrastructure between the cloud and the edge, incorporating devices that can process and store data at multiple points in the network.

Advantages: decentralized, lower latency, localized processing.\
It is ideal for real-time, latency-sensitive applications.

<figure><img src="/assets/image (63).png" alt="" width="563"><figcaption></figcaption></figure>

## Machine (Deep) Learning as a service

The IT architecture for ML/DL on datacenters:

* **Computing cluster**:
  * Servers: parallelization and scalability are key to find the best model as well as Hw accelerators (general purpose CPU and GPU).
  * Storage: multiple storage systems including distributed/parallel file systems (DAS, NAS, SAN).
  * Network: applications are generally non-iterative (ethernet).
* **Virtual Machine Manager**: Virtualization is carried out through hypervisor or containers. Resources are increased by adding more VM, provided that hardware resources are available. User can design personalized software environments and scale instances to the needs.
* **Computing frameworks**: are composed by several modules such as cluster manager, data storage, data processing engine, graph computation, programming languages. An application operating in a cluster is distributed among different computing VM.
* **ML framework**: cover a variety of learning methods for classification, regression, clustering, anomaly detection, and data preparation, and it may or may not include neural network methods.
* **DL framework**: cover a variety of neural network topologies with many hidden layers.

Cloud computing simplifies the access to ML capabilities for designing a solution (without requiring a deep knowledge of ML) and setting up a project (managing demand increases and IT solution).

<figure><img src="/assets/image (154).png" alt="" width="563"><figcaption></figcaption></figure>

## Methods

## Dependability

Dependability is a measure of **how much** we **trust a system**. It is the ability of a system to perform its functionality while exposing:

* **Reliability**: continuity of correct service (affidabilità).
* **Availability**: readiness of correct service (disponibilità). It is ready when needed.
* **Maintainability**: ability for easy maintenance. It is simple to repair.
* **Safety**: absence of catastrophic consequences.
* **Security**: confidentiality and integrity of data.

A lot of effort is devoted to make sure that the implementation matches specifications, fulfills requirements, meets constraints and optimizes selected parameters (performance, energy/power consumption, ..). Nevertheless, even if all above aspects are satisfied, things may fails because something broke.

<figure><img src="/assets/image (155).png" alt="" width="375"><figcaption></figcaption></figure>

A single system failure may affect a large number of people (safety critical systems). A faliures may have high costs if it impacts economic losses or physical damage (mission critical systems). Systems that are not dependable are likely not to be used or adopted (reputation of a brand). Undependable systems may cause information loss with a high consequent recovery cost.

In every single decision phase and in every moment of the life of the system we have to think about dependability. At design time we have to analyse the system under design, measure dependability properties and modify the design if required. At run time we have to detect malfunctions, understand causes and react accordingly.

Failures occur in both development & operation: failures in development should be avoided, on the contrary the ones in operation cannot be avoided and they must be dealt with. Design should take failures into account and guarantee that control and safety are achieved when failures occur. Effect of such failures should be predictable and deterministic, not catastrophic.

Where to apply dependability:

* **Non-critical systems**: a failure during operation can have economic and reputation effects (consumer products).
* **Mission-critical systems**: a failure during operation can have serious or irreversible effect on the mission of the system is carrying out (satellites, automatic weather stations, ..).
* **Safety-critical system**: a failure during operation can present a direct threat to human life (aircraft control systems, medical instrumentation, railway signaling, ..).

**Downtime** is the enemy of every datacenter. Everything has to work properly for the overall system to be working.

How to provide dependability.

* **Failure avoidance paradigm**: conservative design, design validation, detailed test (HW & SW), infant mortality screen, error avoidance.
* **Failure tolerance paradigm**: error detection/masking during system operation, online monitoring, diagnostics, self-recovery & self-repair.

The key elements to provide dependability are a robust design (error-free, trough processes and design practices) and robust operation (fault tolerant, trough monitoring, detection and mitigation).

Safety critical system includes all of the components that work together to achieve the safety-critical mission: they may include input sensors, digital data devices, hardware, peripherals, drivers, actuators, the controlling software, and other interfaces. Their development requires rigorous analysis and comprehensive design and test.

* At the technological level we have to design and manufacture by employing reliable/robust components. Highest dependability, high cost, bad performance.
* At the architecture level ve have to integrate components using solutions that allow to manage the occurrence of failures. High dependability, high cost, reduced performance (depending on the adopted solution).
* At software/application level we have to develop solutions in the algorithms or in the operating systems that mask and recover from the occurrence of failures. High dependability, high cost, reduced performance (depending on the adopted solution).

All of the above solution have in common cost and reduced performance: we have TO PAY for dependability.

The main challenge is to design robust system from unreliable cost Commercial Off-The-Shelf (COTS) components. We have to integrate COTS to get a complex functionality. We have to take into account new problems introduced by technological advances: process variations, stressed working conditions, and the fact that with small geometries, several failure mechanism are becoming now visible at the system-level.

We have to find the best **trade-off** between **dependability** and **costs** depending on

* **Application field.**
* **Working scenario.**
* **Employed technologies.**
* **Algorithms and applications.**

### **Reliability**

It is the ability of a system or component to **perform** its **required functions** under stated conditions for a specified period of time.

$$R(t)$$ is the probability that the system will operate correctly until time t, so $$R(t) = P(not\ failed\ during\ [0, t] )$$ assuming that it was operating a time $$t = 0$$.

Characteristics:

1. **Unreliability**: $$Q(t) = 1- R(t)$$.
2. R(t) is a non-increasing function varying from 1 to 0 over $$[0, +inf )$$ .

It is often used to characterize systems in which even small periods of incorrect behavior are unacceptable: performance requirements, timing requirements, extreme safety requirements, impossibility or difficulty to repair.

### Availability

The the degree to which a system or component is operational and accessible when required for use.\
$$Availability = Uptime / (Uptime + Downtime)$$.

$$A(t)$$ is the probability that the system will be operational at time t.\
$$A(t) = P(not\ failed\ at\ time\ t)$$.

Literally, it is **readiness of service**. It admits the possibility of brief outages. It is fundamentally different from reliability.

Characteristics:

1. **Unavailability**: $$1 - A(t)$$. When the system is not repairable: $$A(t) = R(t)$$. In general (**repairable systems**): $$A(t) ≥ R(t)$$.
2.  Some reference numbers:

    <div><figure><img src="/assets/image (158).png" alt=""><figcaption></figcaption></figure> <figure><img src="/assets/image (159).png" alt=""><figcaption></figcaption></figure></div>

### R(t) & A(t) related indices

* **MTTF** (Mean Time To Failure): mean time before any failure will occur. $$MTTF = \int_{0}^{\inf}{R(t)dt}$$.
*   **MTBF** (Mean Time Between Failures): mean time between two consecutive failures. $$MTBF = total\ operating\ time / number\ of\ failures$$.

    <figure><img src="/assets/image (161).png" alt="" width="375"><figcaption></figcaption></figure>
*   **FIT**: failures in time. It is another way of reporting MTBF. It is the number of expected failures per one billion hours ($$10^9$$) of operation for a device. $$1/MTBF = FIT = number\ of\ failures / total\ operating\ time$$. $$MTBF (in h) = 10^9 / FIT = 1 / \lambda$$ .

    <figure><img src="/assets/image (163).png" alt="" width="375"><figcaption></figcaption></figure>



    * **Infant mortality**: failures showing up in new systems. Usually this category is present during the **testing phases**, and not during production phases.
    * **Random failures**: showing up randomly during the entire life of a system (our main focus).
    * **Wear out**: at the end of its life, some components can cause the failure of a system. Preemptive maintenance can reduce the number of this type of failures. In order to identify defective products and calculate MTTF we can perform the burn-in test: stress the system with excessive temperature, voltage, current, humidity so to accelerate wear out.

    Reliability and availability are two different points of view: the first “the system does not break down”, while the second adds “even if it breaks down, it is working when needed”. They are related: if a system is unavailable it is not delivering the specified system services. It is possibile to have systems with low reliability that must be available: system failures can be repaired quickly and do not damage data, low reliability may not be a problem (for example DBMS). The opposite is in general more difficult.

    Exploitation of R(t) information is used to compute, for a complex system, its reliability in time, that is the expected lifetime (computation of MTBF). Computation of the overall reliability starts from the components’ one.

<figure><img src="/assets/image (164).png" alt="" width="375"><figcaption></figcaption></figure>

#### Reliability block diagram

Reliability block diagrams are an inductive model where a **system** is divided **into blocks** that represent distinct elements such as components or subsystems. Every element in the RBD has its own reliability (previously calculated or modeled). Blocks are then combined together to model all the possibile success paths.

<figure><img src="/assets/image (165).png" alt="" width="563"><figcaption></figcaption></figure>

RDBs are an approach to compute the reliability of a system starting from the reliability of its components.

<figure><img src="/assets/image (166).png" alt="" width="563"><figcaption><p>In series all components must be healthy for the system to work properly. In parallel if one component is healthy the system works properly.</p></figcaption></figure>

In general, if system S is composed by components with a reliability having an exponential distribution (very common case): $$R_s(t) = e^{-\lambda_st}$$ where $$\lambda_s = \sum_{i = 1}^n \lambda_i$$ (failure in time). Then $$MTTF_s = { 1 \over \lambda_s} = 1 / (\sum_{i = 1}^n \lambda_i) = 1 / (\sum_{i = 1}^n {1 \over MTTF_i})$$.

Special case: when all components are identical $$R_s(t) = e^{-\lambda_s t} = e^{-nt / MTTF_1}$$ where $$MTTF_s = MTTF_1 / n$$.

#### Stand-by redundancy

A system may be composed of two parallel replicas:

* The **primary replica** working all time.
* The **redundant replica** (generally not active) that is activated when the primary replica fails.

<figure><img src="/assets/image (168).png" alt="" width="375"><figcaption></figcaption></figure>

Obviously we need:

* A mechanism to determine whether the primary replica is working properly or not (on-line self check).
* A dynamic switching mechanism to disable the primary replica and activate the redundant one.

<figure><img src="/assets/image (169).png" alt="" width="563"><figcaption></figcaption></figure>

More in general, a system having one primary replica and **n redundant replicas** (with **identical** replicas and **perfect switching**). $$R(t) = e^{-\lambda t} \sum_{i = 0}^{n - 1} {{\lambda t}^ i \over i!}$$.

<figure><img src="/assets/image (170).png" alt="" width="375"><figcaption></figcaption></figure>

We can also have a system composed of n identical replicas where at least r replicas have to work fine for the entire system to work fine. $$R(t) = RV \sum_{i = r}^n R_c^i (1 - R_C)^{n - i} {n! \over i! (n-i)!} = RV \sum_{i = r}^n R_c^i (1 - R_C)^{n - i} \binom n i$$

#### Triple Modular Redundancy (TMR)

The system works properly if 2 out of 3 components work properly and the voter works properly. $$R_{TMR} = R_v(3R_m^2 - 2R_m^3)$$ and $$MTTF_{TMR} = {5 \over 6} MTTF_{simplex}$$.

* $$MTTF_{TMR}$$ is shorter than $$MTTF_{simplex}$$.
* It can tolerate transient faults and permanent faults.
* It has higher reliability (for shorter missions).

## Performance Modeling

<figure><img src="/assets/image (171).png" alt="" width="563"><figcaption></figcaption></figure>

Computer performance is the total effectiveness of a computer system, including throughput, individual response time and availability. It can be characterized by the amount of useful work accomplished by a computer system/network compared to the time and resources used.

It is a common practice that system are mostly validated versus “functional” requirements rather than versus quality ones. Different (and often not available) skills are required for quality verification. Short time to market (quickly available products/infrastructures) “seem” to be more attractive nowadays. Little information related to quality is usually available early in the system lifecycle but its understanding is of great importance from the cost and performance point of view (during design, system sizing and system evolution).

System quality can be evaluated trough:

* Use of **intuition** and **trend extrapolation**. In general, those who possess there qualities in sufficient quantity are rare. Pro: **rapid** and **flexible**. Con: accuracy.
* **Experimental evaluation** of **alternatives**. Experimentation is always valuable, often required, and sometimes the approach of choice. It is expensive. An experiment is likely to yield accurate knowledge of system behavior under one set of assumptions, but not any insight that would allow generalization. Pro: **excellent accuracy**. Con: laborious and inflexible.

Complex system can be generalized with a model. Often models are the only artifact to deal with. Model used to drive design decisions. This is the **model-based approach**.

<figure><img src="/assets/image (173).png" alt="" width="563"><figcaption><p>Quality evaluation techniques</p></figcaption></figure>

A model is a representation of a system that is simpler than the actual system, captures the essential (relevant) characteristics and can be evaluated to make predictions.

<figure><img src="/assets/image (174).png" alt="" width="563"><figcaption></figcaption></figure>

### Model-based approach

* **Analytical and numerical techniques** are based on the application of mathematical techniques, which usually exploit results coming from the theory of probability and stochastic process. They are the most efficient and the most precise, but are available only in very limited cases.
* **Simulation techniques** are based on the reproduction of traces of the model. They are the most general, but might also be the less accurate, especially when considering cases in which rare events can occur. The solution time can also become really large when high accuracy is desired.
* **Hybrid techniques** combined analytical/numerical methods with simulation.

### Queueing theory

Queueing theory is the theory behind what happens when you have a **lot of jobs**, **scarce** **resources** and so **long queue** and **delays**. Queueing network modelling is a particular approach in which the **computer system** is represented as a **network of queues**. A network of queues is a collection of service centers, which represent system resources, and customers, which represent user or transactions.

Queueing theory applies whenever queues come up. In a computer system we can have different queues:

* CPU uses a time-sharing scheduler.
* Disks serves a queue of requests waiting to read or write blocks.
* A router in a network serves a queue of packets waiting to be routed.
* Databases have lock queues, where transactions wait to acquire the lock on a record.

<figure><img src="/assets/image (175).png" alt="" width="375"><figcaption></figcaption></figure>

Queueing theory is built on an area of mathematics called stochastic modelling and analysis. Success of queueing network: low-level details of a system are largely irrelevant to its high-level performance characteristics.

The basic scenario for a single queue is that customers, who belong to some population arrive at the service facility. The service facility has one or more servers which can perform the service required by customers. If a customer cannot gain access to a server it must join a queue, in a buffer, until a server is available. When service is complete the customer departs, and the server selects the next customer from the buffer according to the service discipline (queueing policy).

#### Arrival of customers

Arrivals represent jobs entering the system: they specify how fast, how often and which types of jobs does the station service. We are interested in the average arrival rate $$\lambda$$ (req/s).

<figure><img src="/assets/image (176).png" alt="" width="375"><figcaption></figcaption></figure>

#### Service

The service part represents the time a job spends being served. The service time is the time that a server spends satisfying a customer. As with the inter-arrival time, the important characteristics of this time will be its average duration (advance: the distribution function). If the average duration of an interaction between a server and a customer is $$1/\mu$$ then $$\mu$$ is the maximum service rate. We can have:

* **Single server**: the service facility can serve one customer at a time, waiting customers will stay in the buffer until chosen for service, how the next customer is chosen will depend on the service discipline.
* **Multiple server**: there are a fixed number c of servers, each of which can serve a customer. If the number of customers in the facility is less than or equal to c there will no queueing, so each customer will have direct access to a server. If there are more than c customers, the additional customers will have to wait in the buffer.
* **Infinite server**: there are always at least as many servers as there are customers, so that each customer can have a dedicated server as soon as it arrives in the facility. There is no queueing (and no buffer) in such facilities.

#### Queue

If jobs exceed the capacity of parallel processing of the system, they are forced to wait queueing in a buffer. Customers who cannot receive service immediately must wait in the buffer until a server becomes available. If the buffer has finite capacity there are two alternatives for when the buffer becomes full:

* Arrivals are suspended until the facility has space capacity (a customer leaves).
* Arrivals continue and arriving customers are lost until the facility has spare capacity again.

If the buffer capacity is so large that it never affects the behavior of the customers it is assumed to be infinte. Service discipline/queuing policy determines which of the job in the queue will be selected to start its service. The most common used are:

* **FCSF** first-come-first-serve (or FIFO first-in-first-out).
* **LCFS** last-come-first-serve (or LIFO last-in-first-out).
* **RSS** random-selection-for-service.
* **PRI** priority, the assignment of different priorities to elements of a population is one way in which classes are formed.

#### Population

If the size of the population is fixed (N), no more than N customers will ever be requiring service at any time. When the population is finite, the arrival rate of the customers will be affected by the number who are already in the service facility. \
When the size of the population is so large that there is no perceptible impact on the arrival process, we assume that the population is infinite. \
Ideally, members of the population are indistinguishable from each other. When this is not the case, we divide the population into classes whose members all exhibit the same behavior. Different classes differ in one of more characteristics. Identifying classes is a workload characterization task.

In many cases we can see the system as a collection of resources and devices with customers or jobs circulating between them. We can associate a service center with each resource in the system and the route customers among the service center. After service, at one service centre a customer may progress to other service centres, following some previously defined pattern of behaviour, corresponding to the customer’s requirement.

<figure><img src="/assets/image (177).png" alt="" width="563"><figcaption></figcaption></figure>

A queueing network can be represented as a graph where nodes represent the service centers k and arcs the possible transitions of users from one service to another. Nodes and arcs together define the network topology.&#x20;

<figure><img src="/assets/image (178).png" alt="" width="375"><figcaption></figcaption></figure>

A network may be:

* **Open**: customers may arrive from, or depart to, some external environment. They are characterized by arrivals and departures form the system.
* **Closed**. a fixed population of customers remain within the system. We have a parameter N that accounts for the fixed population of jobs that continuously circulate inside the system.
* **Mixed**: there are classes of customers within the system exhibiting open and closed patterns of behaviour respectively.

#### Routing

Whenever a job has alternative routes, an appropriate selection policy must be defined (routing). Routing specification is required only in all the points where jobs exiting a station can have more than one destination. The main routing algorithms we consider are:

* **Probabilistic**: each path has assigned a probability of being chosen by the job that left the considered station.
* **Round robin**: the destination chosen rotates among all the possible exit.
* **Join the shortest queue**: jobs can query the queue length of the possible destinations, and chose to move to the one with the smallest number of jobs waiting to be served

#### Open networks

A client-server system, dealing with external arrivals, which is architected with three tiers.

<figure><img src="/assets/image (180).png" alt="" width="375"><figcaption></figcaption></figure>

<figure><img src="/assets/image (179).png" alt="" width="563"><figcaption><p>Equivalent open network systems</p></figcaption></figure>

Tandem queuing networks are used to model production lines, where raw parts enter the systems, and after a set of stages, the final product is completed (and leaves).

#### Closed networks

A client-server system, with a finite number of customers, which is architected with three tiers.

<figure><img src="/assets/image (181).png" alt="" width="375"><figcaption></figcaption></figure>

<figure><img src="/assets/image (182).png" alt="" width="401"><figcaption></figcaption></figure>

### Operational laws

**Operational laws** are simple **equations** which may be used as an abstract representation or model of the average behaviour of almost any system. The laws are very **general** and make almost **no assumptions** about the behaviour of the random variables characterizing the system, are **simple** and can be applied **quickly** and **easily**. They are based on **observable variables**, values which we could derive from watching a system over a finite period of time. We assume that the system receives requests from its environment. Each request generate a job or a customer within the system. When the job is processed, the system responds to the environment with the result of the request.

<figure><img src="/assets/image (31).png" alt="" width="563"><figcaption><p>measurable quantities</p></figcaption></figure>

<figure><img src="/assets/image (32).png" alt="" width="563"><figcaption><p>derivable quantities</p></figcaption></figure>

We assume that the system is **job flow balanced**: the number of arrivals is equal to the number of completions during an observation period ($$A=C$$). It is a testable assumption since it can be strictly satisfied by careful choice of measurement intervals. If the system is job flow balanced, the arrival rate will be the same as the completion rate ($$\lambda = X$$).

A system may be regarded as being made up of a number of devices or resources. Each of there may be treated as a system in its own right. An external request generates a job within the system, this job may then circulate between the resources until all necessary processing has been done.

<figure><img src="/assets/image (33).png" alt="" width="563"><figcaption></figcaption></figure>

* **Utilization law**: $$U_k = X_k S_k$$, where $$X_k$$ is the number of requests over time (throughput), and $$S_k$$ is the average service time. Assuming that each time a job visit the k-th resource the amount of processing (service time) is $$S_k$$. **Service time** is not necessarily the same as the response time of the job at that resource: in general a job might have to wait for some time before processing begin. The total amount of service that a system job generates at the k-th resources is called the **service demand** $$D_k=S_k V_k$$.
  * Average service time $$S_k$$ accounts for the average time that a job spends in station k when it is served.
  * Average service demand $$D_k$$ accounts for the average time a job spends in station k during its staying in the system. The demand can be greater, less than or equal to the average service depending on the way in which the jobs move in the system.
* **Little’s law**: $$N = XR$$, where $$N$$ is the average number of requests in the system and $$X$$ is the throughput. It can be applied both at system or resource level

<figure><img src="/assets/image (34).png" alt="" width="563"><figcaption></figcaption></figure>

Back when most processing was done on shared mainframes the **think time** was the time that a programmer spent thinking before submitting another job. The think time is the time between processing being completed and the job becoming available as a request again. More generally in interactive systems, job spend time in the system not engaged in processing or waiting for processing.

* **Interactive response time law**: the response time in a interactive system is the residence time minus the think time $$R = N/X - Z$$. Notice that if the think time is zero $$Z = 0$$ then the interactive response time law simply becomes Little’s law.

In an observation interval we can count not only completions external to the system, but also the number of completions at each resource within the system. Let $$C_k$$ be the number of **completions** at resource k. Let $$V_k$$ be the **visit count** of the k-th resource to be the ratio of the number of completions at that resource to the number of system completions $$V_k = C_k / C$$.

* If $$C_k > C$$, resource k is visited several times (on average) during each system level request. This happens when there are loops in the model.
* If $$C_k < C$$, resource k might not be visited during each system level request. This can happen if there are alternatives.
* If $$C_k = C$$, resource k is visited (on average) exactly once every request.
* **Forced flow law**: it captures the relationship between the different components within a system. It states that the trough-puts (flows), in all parts of a system must be proportional to one another $$X_k = V_k X$$. The throughput at the k-th resource is equal to the product of the throughput of the system and the visit count at that resource.

**Residence time** $$\tilde{R_k}$$ accounts for the average time spent in station k, when the job enters the corresponding node. **Response time** $$R_k$$ accounts instead for the average time spent by a job at station k during the staying in the system: it can be greater, small or equal to the response time depending on the number of visits. Relation between the two: $$R_k = v_k \tilde{R_k}$$. So, for a single queue open system (or tandem models $$v_k = 1$$ and $$R_k = \tilde{R_k}$$).

* **General response time law**: another method (than Little’s law) for computing the mean response time per job in a system when N (number of jobs in the system) or X (troughput of the system) are not known. $$R = \sum_k{v_k\tilde{R_k}} = \sum_k{R_k}$$ with $$R_k = v_k\tilde{R_k}$$ $$\forall k$$. The average response time for a job in the system is the sum of the product of the average time for the individual access at each resource and the number of visits it makes to that resource. This is equal to the sum of the resources residence time.

### Performance bounds

The goal is to provide insight into the primary factors affecting the performance of a computer system. They can be computed quickly and easily. We will consider single class system only. We will determine asymptotic bounds (upper and lower bounds) on a system’s performance indices X and R.

The advantage of **bounding analysis** is that it highlight and quantify the critical influence of the system **bottleneck**, it is useful in system sizing and useful for system upgrades. The bottleneck is the resource within a system which has the greatest service demand $$D_{max}$$. It is important because it limits the possibile performance of the system. This will be the resource which has the highest utilization in the system.

<figure><img src="/assets/image (35).png" alt="" width="563"><figcaption></figcaption></figure>

The asymptotic bounds are derived by considering the (asymptotically) extreme conditions of light and heavy loads (**optimistic**: **X upper bound** and **R lower bound**). We work under the assumption that the service demand of a customer at a center does not depend on how many other customers currently are in the system, or at which service centers they are located.

In **open models** we have less information than in closed ones. The X bound is the maximum arrival rate that the system can process. $$\lambda > X_{bound}$$ the system saturates: new jobs have to wait an indefinitely long time. So $$X_{bound} = 1/D_{max}$$ (from the fact that $$U_{max} = \lambda D_{max} ≤ 1$$). The R bounds are the largest and smallest possible R experienced at a given $$\lambda$$ investigate only when $$\lambda < \lambda_{sat}$$ (otherwise the system is unstable). If no customer interferes with any other (no queue time) then $$R = D = \sum_k D_k$$. In open models there is no pessimistic bound on R: we have batch of n request at the same time, customers at the end of the batch are forced to queue for customers at the front and thus experience large response time. In this case response time increases with the increasing of the batch size. In the worst case the response time is infinity and we cannot say more.

In **closed models** we consider X bounds first and then convert it into R bounds using Little’s law. The highest possibile system response time occurs when each job, at each iteration, founds all the other N-1 costumers in front of it. The lowest response time can be obtained if a job always find the queue empty and always start being served immediately.

In conclusion: $$\frac{N}{ND + Z} ≤ X(N) ≤ min\{1/D_{max}, \frac{N}{D+Z}\}$$ with $$N^* = \frac{D+Z}{D_{max}}$$.

<figure><img src="/assets/image (36).png" alt="" width="563"><figcaption></figcaption></figure>

For R bounds we simply rewrite the previous equation considering $$X(N) = \frac{N}{R(N)+Z}$$ and we obtain $$max\{D, ND_{max}-Z\} ≤ R(N) ≤ ND$$.

