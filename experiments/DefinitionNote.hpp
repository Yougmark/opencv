template <typename T>
using QueueClass =
    tbb::concurrent_bounded_queue<T>; // src/executor/gstreamingexecutor.hpp
namespace cv
{

/* -- include/opencv2/gapi/gcommon.hpp */
enum class GShape : int
{
    GMAT,
    GSCALAR,
    GARRAY,
    GOPAQUE,
};
struct GAPI_EXPORTS GCompileArg
{
public:
    std::string tag;
    // FIXME: use decay in GArg/other trait-based wrapper before leg is shot!
    template <typename T, typename std::enable_if<
                              !detail::is_compile_arg<T>::value, int>::type = 0>
    explicit GCompileArg(T &&t)
        : tag(detail::CompileArgTag<typename std::decay<T>::type>::tag()),
          arg(t)
    {
    }
    template <typename T> T &get() { return util::any_cast<T>(arg); }
    template <typename T> const T &get() const
    {
        return util::any_cast<T>(arg);
    }

private:
    util::any arg;
};
using GCompileArgs = std::vector<GCompileArg>;
/**
 * Wraps a list of arguments (a parameter pack) into a vector of
 * compilation arguments (cv::GCompileArg).
 */
template <typename... Ts> GCompileArgs compile_args(Ts &&... args)
{
    return GCompileArgs{ GCompileArg(args)... };
}
/* include/opencv2/gapi/gcommon.hpp -- */

/* -- src/include/opencv2/gapi/garg.hpp */
class GAPI_EXPORTS GArg
{
public:
    GArg() {}
    template <typename T, typename std::enable_if<!detail::is_garg<T>::value,
                                                  int>::type = 0>
    explicit GArg(const T &t)
        : kind(detail::GTypeTraits<T>::kind),
          value(detail::wrap_gapi_helper<T>::wrap(t))
    {
    }
    template <typename T, typename std::enable_if<!detail::is_garg<T>::value,
                                                  int>::type = 0>
    explicit GArg(T &&t)
        : kind(detail::GTypeTraits<typename std::decay<T>::type>::kind),
          value(detail::wrap_gapi_helper<T>::wrap(t))
    {
    }
    template <typename T> inline T &get()
    {
        return util::any_cast<typename std::remove_reference<T>::type>(value);
    }
    template <typename T> inline const T &get() const
    {
        return util::any_cast<typename std::remove_reference<T>::type>(value);
    }
    template <typename T> inline T &unsafe_get()
    {
        return util::unsafe_any_cast<typename std::remove_reference<T>::type>(
            value);
    }
    template <typename T> inline const T &unsafe_get() const
    {
        return util::unsafe_any_cast<typename std::remove_reference<T>::type>(
            value);
    }
    detail::ArgKind kind = detail::ArgKind::OPAQUE_VAL;

protected:
    util::any value;
};
using GRunArgP = util::variant<cv::UMat *, cv::Mat *, cv::Scalar *,
                               cv::detail::VectorRef, cv::detail::OpaqueRef>;
using GRunArg = util::variant<
#if !defined(GAPI_STANDALONE)
    cv::UMat,
#endif // !defined(GAPI_STANDALONE)
    cv::gapi::wip::IStreamSource::Ptr, cv::Mat, cv::Scalar,
    cv::detail::VectorRef, cv::detail::OpaqueRef>;
/* src/include/opencv2/gapi/garg.hpp -- */

using GMetaArg = util::variant<util::monostate, GMatDesc, GScalarDesc,
                               GArrayDesc, GOpaqueDesc>;
using GMetaArgs = std::vector<GMetaArg>;

namespace gapi
{
/* -- include/opencv2/gapi/gkernel.hpp */
class GAPI_EXPORTS GBackend
{
public:
    class Priv;
    GBackend();
    explicit GBackend(std::shared_ptr<Priv> &&p);
    Priv &priv();
    const Priv &priv() const;
    std::size_t hash() const;
    bool operator==(const GBackend &rhs) const;

private:
    std::shared_ptr<Priv> m_priv;
};
class GFunctor
{
public:
    virtual cv::GKernelImpl impl() const = 0;
    virtual cv::gapi::GBackend backend() const = 0;
    const char *id() const { return m_id; }

    virtual ~GFunctor() = default;

protected:
    GFunctor(const char *id) : m_id(id){};

private:
    const char *m_id;
};
/* include/opencv2/gapi/gkernel.hpp -- */
} // namespace gapi
namespace gimpl
{

/* -- src/modules/gapi/src/backends/commons/gbackend.hpp */
namespace magazine
{
template <typename... Ts> struct Class
{
    template <typename T> using MapT = std::unordered_map<int, T>;
    template <typename T> MapT<T> &slot()
    {
        return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
    }
    template <typename T> const MapT<T> &slot() const
    {
        return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
    }

private:
    std::tuple<MapT<Ts>...> slots;
};

} // namespace magazine
#if !defined(GAPI_STANDALONE)
using Mag = magazine::Class<cv::Mat, cv::UMat, cv::Scalar,
                            cv::detail::VectorRef, cv::detail::OpaqueRef>;
#else
using Mag = magazine::Class<cv::Mat, cv::Scalar, cv::detail::VectorRef,
                            cv::detail::OpaqueRef>;
#endif

namespace magazine
{
// copy input data from arg to Mag
// note that GRunArg is a variant (cv::util::variant), and Mag is a struct with
// one map<int, type> per type.  The shape (for type) and id (for int) in RcDesc
// together locates the data in Mag.
void GAPI_EXPORTS bindInArg(Mag &mag, const RcDesc &rc, const GRunArg &arg,
                            bool is_umat = false);
// copy pointer of output buffer from arg to Mag, with read and write permission
void GAPI_EXPORTS bindOutArg(Mag &mag, const RcDesc &rc, const GRunArgP &arg,
                             bool is_umat = false);
void resetInternalData(Mag &mag, const Data &d);
// create GRunArg with data located by ref in mag.
cv::GRunArg getArg(const Mag &mag, const RcDesc &ref);
// similar as getArg, but create GRunArgP, which is a variant of pointers.
cv::GRunArgP getObjPtr(Mag &mag, const RcDesc &rc, bool is_umat = false);
// (?) don't understand what writeBack does
void writeBack(const Mag &mag, const RcDesc &rc, GRunArgP &g_arg,
               bool is_umat = false);
} // namespace magazine
struct GRuntimeArgs
{
    GRunArgs inObjs;
    GRunArgsP outObjs;
};
// get the compile argument of the type T.
template <typename T>
inline cv::util::optional<T> getCompileArg(const cv::GCompileArgs &args);
// create a Mat using mat.create with dimensions described in desc.
void createMat(const cv::GMatDesc &desc, cv::Mat &mat);
/* src/modules/gapi/src/backends/commons/gbackend.hpp -- */

/*************************************************
 ****************** compiler *********************
 ************************************************/

/* -- src/compiler/gobjref.hpp */
using HostCtor = util::variant<util::monostate, detail::ContructVec,
                               detail::ConstructOpaque>;
using ConstVal = util::variant<util::monostate, cv::Scalar>;
struct RcDesc
{
    int id;       // id is unique but local to shape
    GShape shape; // pair <id,shape> IS the unique ID
    HostCtor ctor;
    bool operator==(const RcDesc &rhs) const
    {
        return id == rhs.id && shape == rhs.shape;
    }
    bool operator<(const RcDesc &rhs) const
    {
        return (id == rhs.id) ? shape < rhs.shape : id < rhs.id;
    }
};
/* src/compiler/gobjref.hpp -- */

/* -- src/compiler/gmodel.hpp */
struct NodeType
{
    static const char *name() { return "NodeType"; }
    enum
    {
        OP,
        DATA
    } t;
};
struct Input;
struct Output;
struct Op
{
    static const char *name() { return "Op"; }
    cv::GKernel k;
    std::vector<GArg> args;   // TODO: Introduce a new type for internal args?
    std::vector<RcDesc> outs; // TODO: Introduce a new type for resource
                              // references
    cv::gapi::GBackend backend;
};
struct Data
{
    static const char *name() { return "Data"; }

    // FIXME: This is a _pure_ duplication of RcDesc now! (except storage)
    GShape shape; // FIXME: Probably to be replaced by GMetaArg?
    int rc;
    GMetaArg meta;
    HostCtor ctor; // T-specific helper to deal with unknown types in our code
    // FIXME: Why rc+shape+meta is not represented as RcDesc here?

    enum class Storage
    {
        INTERNAL,  // data object is not listed in GComputation protocol
        INPUT,     // data object is listed in GComputation protocol as Input
        OUTPUT,    // data object is listed in GComputation protocol as Output
        CONST_VAL, // data object is constant.
                   // Note: CONST is sometimes defined in Win sys headers
    };
    Storage storage;
};
struct ConstValue;
struct Island;
// Ins and Outs for the graph
struct Protocol
{
    static const char *name() { return "Protocol"; }
    std::vector<RcDesc> inputs;
    std::vector<RcDesc> outputs;
    std::vector<ade::NodeHandle> in_nhs;
    std::vector<ade::NodeHandle> out_nhs;
};
struct OriginalInputMeta;
struct OtuputMeta;
struct Journal;
class DataObjectCounter;
// one-island-per-node graph generated from one-operation-per-node graph.
// for the first level of execution
struct IslandModel
{
    static const char *name() { return "IslandModel"; }
    std::shared_ptr<ade::Graph> model;
};
struct ActiveBackends;
// a graph-level flag indicating the graph is compiled for streaming mode
// (pipelined execution).
struct Streaming;
// inference parameters for the iebackend.
struct NetworkParams;
struct CustomMetaFunction;

namespace GModel
{
// how are these types used?
using Graph =
    ade::TypedGraph<NodeType, Input, Output, Op, Data, ConstValue, Island,
                    Protocol, OriginalInputMeta, OutputMeta, Journal,
                    ade::passes::TopologicalSortData, DataObjectCounter,
                    IslandModel, ActiveBackends, CustomMetaFunction, Streaming>;
using ConstGraph =
    ade::ConstTypedGraph<NodeType, Input, Output, Op, Data, ConstValue, Island,
                         Protocol, OriginalInputMeta, OutputMeta, Journal,
                         ade::passes::TopologicalSortData, DataObjectCounter,
                         IslandModel, ActiveBackends, CustomMetaFunction,
                         Streaming>;
} // namespace GModel
/* src/compiler/gmodel.hpp -- */

/* -- src/compiler/gislandmodel.hpp */
class GIsland
{
public:
    using node_set =
        std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>>;
    GIsland(const gapi::GBackend &bknd, ade::NodeHandle op,
            util::optional<std::string> &&user_tag);
    GIsland(const gapi::GBackend &bknd, node_set &&all, node_set &&in_ops,
            node_set &&out_ops, util::optional<std::string> &&user_tag);
    const node_set &contents() const;
    const node_set &in_ops() const;
    const node_set &out_ops() const;
    std::string name() const;
    gapi::GBackend backend() const;
    node_set consumers(const ade::Graph &g,
                       const ade::NodeHandle &slot_nh) const;
    ade::NodeHandle producer(const ade::Graph &g,
                             const ade::NodeHandle &slot_nh) const;
    void debug() const;
    bool is_user_specified() const;

protected:
    gapi::GBackend m_backend; // backend which handles this Island execution
    node_set m_all;           // everything (data + operations) within an island
    node_set m_in_ops;        // operations island begins with
    node_set m_out_ops;       // operations island ends with
    // has island name IF specified by user. Empty for internal (inferred)
    // islands
    util::optional<std::string> m_user_tag;
};

class GIslandExecutable
{
public:
    using InObj = std::pair<RcDesc, cv::GRunArg>;
    using OutObj = std::pair<RcDesc, cv::GRunArgP>;
    class IODesc;
    struct IInput;
    struct IOutput;
    virtual void run(std::vector<InObj> &&input_objs,
                     std::vector<OutObj> &&output_objs) = 0;
    virtual void run(IInput &in, IOutput &out);
    virtual bool canReshape() const = 0;
    virtual void reshape(ade::Graph &g, const GCompileArgs &args) = 0;
    virtual void handleNewStream(){}; // do nothing here by default
    virtual ~GIslandExecutable() = default;
};

class GIslandExecutable::IODesc;

struct EndOfStream
{
};
using StreamMsg = cv::util::variant<EndOfStream, cv::GRunArgs>;

struct GIslandExecutable::IInput : public GIslandExecutable::IODesc
{
    // get a new input vector (blocking)
    virtual StreamMsg get() = 0;
    // get a new input vector (non-blocking)
    virtual StreamMsg try_get() = 0;
};

struct GIslandExecutable::IOutput : public GIslandExecutable::IODesc
{
    // allocate a new data object for output idx
    virtual GRunArgP get(int idx) = 0;
    // release the object back to the framework (mark available)
    virtual void post(GRunArgp &&) = 0;
    // release end-of-stream marker back to the framework
    virtual void post(EndOfStream &&) = 0;
};

class GIslandEmitter;
struct NodeKind
{
    static const char *name() { return "NodeKind"; }
    enum
    {
        ISLAND,
        SLOT,
        EMIT,
        SINK
    } k;
};
struct FusedIsland
{
    static const char *name() { return "FusedIsland"; }
    std::shared_ptr<GIsland> object;
};
struct DataSlot
{
    static const char *name() { return "DataSlot"; }
    ade::NodeHandle original_data_node; // direct link to GModel
};
struct IslandExec
{
    static const char *name() { return "IslandExecutable"; }
    std::shared_ptr<GIslandExecutable> object;
};
struct Emitter
{
    static const char *name() { return "Emitter"; }
    std::size_t proto_index;
    std::shared_ptr<GIslandEmitter> object;
};
struct Sink
{
    static const char *name() { return "Sink"; }
    std::size_t proto_index;
};
struct IslandsCompiled
{
    static const char *name() { return "IslandsCompiled"; }
};

namespace GIslandModel
{
using Graph =
    ade::TypedGraph<NodeKind, FusedIsland, DataSlot, IslandExec, Emitter, Sink,
                    IslandsCompiled, ade::passes::TopologicalSortData>;
using ConstGraph =
    ade::ConstTypedGraph<NodeKind, FusedIsland, DataSlot, IslandExec, Emitter,
                         Sink, IslandsCompiled,
                         ade::passes::TopologicalSortData>;
void generateInitial(Graph &g, const ade::Graph &src_g);
ade::NodeHandle mkSlotNode(Graph &g, const ade::NodeHandle &data_nh);
ade::NodeHandle mkIslandNode(Graph &g, const gapi::GBackend &bknd,
                             const ade::NodeHandle &op_nh,
                             const ade::Graph &orig_g);
ade::NodeHandle mkIslandNode(Graph &g, std::shared_ptr<GIsland> &&isl);
ade::NodeHandle mkEmitNode(Graph &g, std::size_t in_idx);  // streaming-related
ade::NodeHandle mkSinkNode(Graph &g, std::size_t out_idx); // streaming-related
void syncIslandTags(Graph &g, ade::Graph &orig_g);
void compileIslands(Graph &g, const ade::Graph &orig_g,
                    const GCompileArgs &args);
ade::NodeHandle GAPI_EXPORTS producerOf(const ConstGraph &g,
                                        ade::NodeHandle &data_nh);
} // namespace GIslandModel
/* src/compiler/gislandmodel.hpp -- */

/*************************************************
 ****************** executor *********************
 ************************************************/

/* -- src/executor/gexecutor.hpp */
class GExecutor
{
protected:
    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;
    cv::gimpl::GModel::Graph m_gm;
    cv::gimpl::GIslandModel::Graph g_gim;
    struct OpDesc
    {
        std::vector<RcDesc> in_objects;
        std::vector<RcDesc> out_objects;
        std::shared_ptr<GIslandExecutable> isl_exec;
    };
    std::vector<OpDesc> m_ops;
    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };
    std::vector<DataDesc> m_slots;
    class Input;
    class Output;
    Mag m_res;
    void initResource(const ade::NodeHandle &orig_nh);

public:
    // prepare island executors (m_ops)
    explicit GExecutor(std::unique_ptr<ade::Graph> &&g_model);
    void run(cv::gimpl::GRuntimeArgs &&args);
    bool canReshape() const;
    void reshape(const GMetaArgs &inMetas, const GCompileArgs &args);
    const GModel::Graph &model() const;
};
/* src/executor/gexecutor.hpp -- */

/* -- src/executor/gstreamingexecutor.hpp */
namespace stream
{
struct Start;
struct Stop;
using Cmd = cv::util::variant<cv::util::monostate, Start, Stop, cv::GRunArg,
                              cv::GRunArgs>;
using Q = QueueClass<Cmd>;
} // namespace stream
class GStreamingExecutor final
{
    enum class State
    {
        STOPPED,
        READY,
        RUNNING,
    } state = State::STOPPED;
    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;
    cv::GCompileArgs m_comp_args;
    cv::GMetaArgs m_last_metas;
    util::optional<bool> m_reshapable;
    cv::gimpl::GIslandModel::Graph m_gim;
    struct OpDesc
    {
        std::vector<RcDesc> in_objects;
        std::vector<RcDesc> out_objects;
        cv::GMetaArgs out_metas;
        ade::NodeHandle nh;
        cv::GRunArgs in_constants;
        std::shared_ptr<GIslandExecutable> isl_exec;
    };
    // one item per island
    std::vector<OpDesc> m_ops;
    struct DataDesc
    {
        ade::NodeHandle slot_nh;
        ade::NodeHandle data_nh;
    };
    std::vector<DataDesc> m_slots;
    cv::GRunArgs m_const_vals;
    std::vector<ade::NodeHandle> m_emitters;
    std::vector<ade::NodeHandle> m_sinks;
    std::vector<std::thread> m_threads;
    std::vector<stream::Q> m_emitter_queues;
    std::vector<stream::Q *> m_const_emitter_queues;
    std::vector<stream::Q *> m_sink_queues;
    std::unordered_set<stream::Q *> m_internal_queues;
    stream::Q m_out_queue;
    void wait_shutdown();

public:
    // prepare m_ops, internal data queues (m_internal_queues), m_slots,
    // m_emitters, m_emitter_queues, m_sinks, m_sink_queues, m_out_queue
    explicit GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model);
    ~GStreamingExecutor();
    // compile islands if not yet
    // set up emitters
    // set up execution threads (m_threads) for emitter and islands.
    void setSource(GRunArgs &&args);
    // push stream::Start{} into m_emitter_queues
    void start();
    bool pull(cv::GRunArgsP &&outs);
    bool try_pull(cv::GRunArgsP &&outs);
    void stop();
    bool running() const;
}
/* src/executor/gstreamingexecutor.hpp -- */

/* -- src/executor/gstreamingexecutor.cpp */
namespace
{
    // read data from video file
    class VideoEmitter final : public cv::gimpl::GIslandEmitter;
    // read const data
    class ConstEmitter;
    // used as type for ade::TypedGraph (?)
    struct DataQueue;
    // get queues for out edges
    std::vector<cv::gimpl::stream::Q *> reader_queues(
        ade::Graph & g, const ade::NodeHandle &obj);
    // get queues for in edges
    std::vector<cv::gimpl::stream::Q *> input_queues(
        ade::Graph & g, const ade::NodeHandle &obj);
    // get output data
    void sync_data(cv::GrunArgs & results, cv::GRunArgsP & outputs);
    class QueueReader
    {
    public:
        // prepare input data
        // sleep if data is not available, at Q::pop().
        bool getInputVector(std::vector<Q *> &in_queues,
                            cv::GRunArgs &in_constants,
                            cv::GRunArgs *isl_inputs)
    };
    // thread for obtaining data from source node
    void emitterActorThread(std::shared_ptr<cv::gimpl::GIslandEmitter> emitter,
                            Q & in_queue, std::vector<Q *> out_queues,
                            std::function<void()> callback_completion);
    class StreamingInput final : public cv::gimpl::GIslandExecutable::IInput
    {
        // uses QueueReader::getInputVector to get next (?) input data
        virtual cv::impl::StreamMsg get() override;
        // current implementation calls get()
        virtual cv::impl::StreamMsg try_get() override;
    };
    class StreamingOutput final : public cv::gimpl::GIslandExecutable::IOutput
    {
        struct Posting
        {
            using V = cv::util::variant<cv::GRunArg, cv::gimpl::EndOfStream>;
            V data;
            bool ready = false;
        };
        using PostingList = std::list<Posting>;
        std::vector<PostingList> m_postings;
        // allocate a new data object for output under idx for posting.
        virtual cv::GRunArgP get(int idx) override;
        virtual void post(cv::GRunArgP &&argp) override;
        virtual void post(cv::gimpl::EndOfStream &&) override;
    };
    // one thread per island (?)
    void islandActorThread(
        std::vector<cv::gimpl::RcDesc> in_rcs,
        std::vector<cv::gimpl::RcDesc> out_rcs, cv::GMetaArgs out_metas,
        std::shared_ptr<cv::gimpl::GIslandExecutable> island,
        std::vector<Q *> in_queues, cv::GRunArgs in_constants,
        std::vector<std::vector<Q *>> out_queues);
    // merge from multiple output queues so that try_pull() can be
    // implemented conveniently by check only one queue.
    void collectorThread(std::vector<Q *> in_queues, Q & out_queue);
} // namespace
/* src/executor/gstreamingexecutor.cpp -- */

/*************************************************
 ****************** backend *********************
 ************************************************/

/* -- src/backends/gcpubackend.hpp */
struct Unit
{
    static const char *name() { return "HostKernel"; }
    GCPUKernel k;
};
class GCPUExecutable final : public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;
    struct OperationInfo
    {
        ade::NodeHandle nh;
        GMetaArgs expected_out_metas;
    };
    std::vector<OperationInfo> m_script;      // one item per node for execution
    std::vector<ade::NodeHandle> m_dataNodes; // all resources in graph (?)
    Mag m_res; // actual data of all resources (?)
    GArg packArg(const GArg &arg);

public:
    GCPUExecutable(const ade::Graph &graph,
                   const std::vector<ade::NodeHandle> &nodes);
    virtual inline bool canReshape() const override;
    virtual inline void reshape(ade::Graph &, const GCompileArgs &) override;
    virtual void run(std::vector<InObj> &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};
/* src/backends/gcpubackend.hpp -- */

/* -- src/backends/gcpubackend.cpp */
namespace
{
class GCPUBackendImpl final : public cv::gapi::GBackend::Priv
{
    // unpackKernel is called during kernel pass
    // (src/compiler/passes/kernels.cpp).
    // This unpackKernel adds the cv::GKernelImpl (cv::GCPUKernel herein) to the
    // graph property (Unit) of the corresponding node handle (op_node).
    virtual void unpackKernel(ade::Graph &graph, const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
        GCPUModel gm(graph);
        auto cpu_impl = cv::util::any_cast<cv::GCPUKernel>(impl.opaque);
        gm.metadata(op_node).set(cv::gimpl::Unit{ cpu_impl });
    }
    virtual EPtr
    compile(const ade::Graph &graph, const cv::GCompileArgs &,
            const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{ new cv::gimpl::GCPUExecutable(graph, nodes) };
    }
};
} // namespace
cv::gapi::GBackend cv::gapi::cpu::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GCPUBackendImpl>());
    return this_backend;
}
// get data from m_res and encompass it in a GArg
cv::GArg cv::gimpl::GCPUExecutable::packArg(const GArg &arg);
/* src/backends/gcpubackend.cpp -- */

} // namespace gimpl

/* -- src/backends/gcpukernel.hpp */
class GAPI_EXPORTS GCPUContext
{
public:
    template <typename T> const T &inArg(int input)
    {
        return m_args.at(input).get<T>();
    }
    const cv::Mat &inMat(int input);
    cv::Mat &outMatR(int output); // FIXME: Avoid cv::Mat m = ctx.outMatR()
    const cv::Scalar &inVal(int input);
    cv::Scalar &outValR(int output); // FIXME: Avoid cv::Scalar s =
    template <typename T>
    std::vector<T> &outVecR(int output) // FIXME: the same issue
    {
        return outVecRef(output).wref<T>();
    }
    template <typename T> T &outOpaqueR(int output) // FIXME: the same issue
    {
        return outOpaqueRef(output).wref<T>();
    }

protected:
    detail::VectorRef &outVecRef(int output);
    detail::OpaqueRef &outOpaqueRef(int output);
    std::vector<GArg> m_args;
    // FIXME: avoid conversion of arguments from internal representation to
    // OpenCV one on each call to OCV kernel. (This can be achieved by a two
    // single time conversions in GCPUExecutable::run, once on enter for input
    // and output arguments, and once before return for output arguments only
    std::unordered_map<std::size_t, GRunArgP> m_results;
    friend class gimpl::GCPUExecutable;
    friend class gimpl::render::ocv::GRenderExecutable;
};
class GAPI_EXPORTS GCPUKernel
{
public:
    // This function is kernel's execution entry point (does the processing
    // work)
    using F = std::function<void(GCPUContext &)>;
    GCPUKernel();
    explicit GCPUKernel(const F &f);
    void apply(GCPUContext &ctx);

protected:
    F m_f;
};
// This class does not have GKernelImpl as in GOCVFunctor.
// But it is added when kernels() is called.
// see `template<typename... KK> GKernelPackage kernels()` in
// include/opencv/gapi/gkernel.hpp
template <class Impl, class K>
class GCPUKernelImpl
    : public cv::detail::OCVCallHelper<Impl, typename K::InArgs,
                                       typename K::OutArgs>,
      public cv::detail::KernelTag
{
    using P =
        detail::OCVCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

public:
    using API = K;
    static cv::gapi::GBackend backend() { return cv::gapi::cpu::backend(); }
    static cv::GCPUKernel kernel() { return GCPUKernel(&P::call); }
};

#define GAPI_OCV_KERNEL(Name, API)                                             \
    struct Name : public cv::GCPUKernelImpl<Name, API>

class gapi::cpu::GOCVFunctor : public gapi::GFunctor
{
public:
    using Impl = std::function<void(GCPUContext &)>;

    GOCVFunctor(const char *id, const Impl &impl)
        : gapi::GFunctor(id), impl_{ GCPUKernel(impl) }
    {
    }

    GKernelImpl impl() const override { return impl_; }
    gapi::GBackend backend() const override { return gapi::cpu::backend(); }

private:
    GKernelImpl impl_;
};

//! @cond IGNORED
template <typename K, typename Callable>
gapi::cpu::GOCVFunctor gapi::cpu::ocv_kernel(Callable &c)
{
    using P = detail::OCVCallHelper<Callable, typename K::InArgs,
                                    typename K::OutArgs>;
    return GOCVFunctor(K::id(), std::bind(&P::callFunctor,
                                          std::placeholders::_1, std::ref(c)));
}

template <typename K, typename Callable>
gapi::cpu::GOCVFunctor gapi::cpu::ocv_kernel(const Callable &c)
{
    using P = detail::OCVCallHelper<Callable, typename K::InArgs,
                                    typename K::OutArgs>;
    return GOCVFunctor(K::id(),
                       std::bind(&P::callFunctor, std::placeholders::_1, c));
}
//! @endcond
/* -- src/backends/gcpukernel.hpp */
} // namespace cv
