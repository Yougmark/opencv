// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "precomp.hpp"

#include <ade/util/algorithm.hpp>

#include <ade/util/chain_range.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>

#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/util/any.hpp>

#include "compiler/gmodel.hpp"
#include "compiler/gobjref.hpp"

#include "backends/rt/grtbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

using GRTModel = ade::TypedGraph<cv::gimpl::Unit>;
using GConstGRTModel = ade::ConstTypedGraph<cv::gimpl::Unit>;

namespace
{
class GRTBackendImpl final : public cv::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph &graph, const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
        GRTModel gm(graph);
        auto rt_impl = cv::util::any_cast<cv::GRTKernel>(impl.opaque);
        gm.metadata(op_node).set(cv::gimpl::Unit{ rt_impl });
    }

    virtual EPtr
    compile(ade::Graph &graph, const cv::GCompileArgs &,
            const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{ new cv::gimpl::GRTExecutable(graph, nodes) };
    }
};
} // namespace

struct DataQueue
{
    static const char *name() { return "StreamingDataQueue"; }

    explicit DataQueue(std::size_t capacity)
    {
        if (capacity)
        {
            q.set_capacity(capacity);
        }
    }

    cv::gimpl::stream::Q q;
};

cv::gapi::GBackend cv::gapi::rt::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GRTBackendImpl>());
    return this_backend;
}

void cv::gimpl::GRTExecutable::nodeActorThread(
    cv::gimpl::GRTExecutable::OperationInfo &op_info)
{
    cv::gimpl::stream::Cmd cmd;
    const auto &op = m_gm.metadata(op_info.nh).get<Op>();

    // get input data queues for this node handle.
    ade::TypedGraph<DataQueue> qgr(m_g);
    for (auto &&in_eh : op_info.nh->inEdges())
    {
        if (qgr.metadata(in_eh).contains<DataQueue>())
        {
            qgr.metadata(in_eh).get<DataQueue>().q.pop(cmd);
        }
    }

    // Obtain our real execution unit
    // TODO: Should kernels be copyable?
    GConstGRTModel gcm(m_g);
    GRTKernel k = gcm.metadata(op_info.nh).get<Unit>().k;

    // Initialize kernel's execution context:
    // - Input parameters
    GRTContext context;
    context.m_args.reserve(op.args.size());

    using namespace std::placeholders;
    ade::util::transform(op.args, std::back_inserter(context.m_args),
                         std::bind(&GRTExecutable::packArg, this, _1));

    // - Output parameters.
    // FIXME: pre-allocate internal Mats, etc, according to the known meta
    for (const auto &out_it : ade::util::indexed(op.outs))
    {
        // FIXME: Can the same GArg type resolution mechanism be reused
        // here?
        const auto out_port = ade::util::index(out_it);
        const auto out_desc = ade::util::value(out_it);
        context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc);
    }

    // Now trigger the executable unit
    k.apply(context);

    // As Kernels are forbidden to allocate memory for (Mat) outputs,
    // this code seems redundant, at least for Mats
    // FIXME: unify with cv::detail::ensure_out_mats_not_reallocated
    // FIXME: when it's done, remove can_describe(const GMetaArg&, const
    // GRunArgP&) and descr_of(const cv::GRunArgP &argp)
    for (const auto &out_it : ade::util::indexed(op_info.expected_out_metas))
    {
        const auto out_index = ade::util::index(out_it);
        const auto expected_meta = ade::util::value(out_it);

        if (!can_describe(expected_meta, context.m_results[out_index]))
        {
            const auto out_meta = descr_of(context.m_results[out_index]);
            util::throw_error(
                std::logic_error("Output meta doesn't "
                                 "coincide with the generated meta\n"
                                 "Expected: " +
                                 ade::util::to_string(expected_meta) +
                                 "\n"
                                 "Actual  : " +
                                 ade::util::to_string(out_meta)));
        }
    }

    // (?) each out edge has inEdges and outEdges?
    // And outEdges are the inEdges of next node?
    for (auto &&out_eh : op_info.nh->outNodes())
    {
        for (auto &&out_eeh : out_eh->outEdges())
        {
            if (qgr.metadata(out_eeh).contains<DataQueue>())
            {
                qgr.metadata(out_eeh).get<DataQueue>().q.push(cmd);
            }
        }
    }
}

void cv::gimpl::GRTExecutable::prepScriptData(
    const std::vector<ade::NodeHandle> &nodes)
{
    ade::TypedGraph<DataQueue> qgr(m_g);
    for (auto &nh : nodes)
    {
        switch (m_gm.metadata(nh).get<NodeType>().t)
        {
        case NodeType::OP:
        {
            m_script.push_back({ nh, GModel::collectOutputMeta(m_gm, nh) });
            for (auto eh : nh->inEdges())
            {
                qgr.metadata(eh).set(DataQueue(10));
                m_internal_queues.insert(&qgr.metadata(eh).get<DataQueue>().q);
            }
            break;
        }
        case NodeType::DATA:
        {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST_VAL)
            {
                auto rc = RcDesc{ desc.rc, desc.shape, desc.ctor };
                magazine::bindInArg(m_res, rc,
                                    m_gm.metadata(nh).get<ConstValue>().arg);
            }
            // preallocate internal Mats in advance
            if (desc.storage == Data::Storage::INTERNAL &&
                desc.shape == GShape::GMAT)
            {
                const auto mat_desc = util::get<cv::GMatDesc>(desc.meta);
                auto &mat = m_res.slot<cv::Mat>()[desc.rc];
                createMat(mat_desc, mat);
            }
            break;
        }
        default:
        {
            util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
        }
    }
    // note that the last operation is executed in run()
    for (auto it = m_script.begin(); it < m_script.end() - 1; it++)
    {
        m_threads.emplace_back(&cv::gimpl::GRTExecutable::nodeActorThread, this,
                               std::ref(*it));
    }
}

// GRTExecutable implementation //////////////////////////////////////////////
cv::gimpl::GRTExecutable::GRTExecutable(
    ade::Graph &g, const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
    prepScriptData(nodes);
}

// FIXME: Document what it does
cv::GArg cv::gimpl::GRTExecutable::packArg(const GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(arg.kind != cv::detail::ArgKind::GMAT &&
                arg.kind != cv::detail::ArgKind::GSCALAR &&
                arg.kind != cv::detail::ArgKind::GARRAY &&
                arg.kind != cv::detail::ArgKind::GOPAQUE);

    if (arg.kind != cv::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg
        // contents.
        return arg;
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated RT object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:
        return GArg(m_res.slot<cv::Mat>()[ref.id]);
    case GShape::GSCALAR:
        return GArg(m_res.slot<cv::Scalar>()[ref.id]);
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be
    // already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:
        return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));
    case GShape::GOPAQUE:
        return GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::GRTExecutable::run(std::vector<InObj> &&input_objs,
                                   std::vector<OutObj> &&output_objs)
{
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto &it : input_objs)
        magazine::bindInArg(m_res, it.first, it.second);
    for (auto &it : output_objs)
        magazine::bindOutArg(m_res, it.first, it.second);

    // Initialize (reset) internal data nodes with user structures
    // before processing a frame (no need to do it for external data structures)
    GModel::ConstGraph gm(m_g);
    for (auto nh : m_dataNodes)
    {
        const auto &desc = gm.metadata(nh).get<Data>();

        if (desc.storage == Data::Storage::INTERNAL &&
            !util::holds_alternative<util::monostate>(desc.ctor))
        {
            // FIXME: Note that compile-time constant data objects (like
            // a value-initialized GArray<T>) also satisfy this condition
            // and should be excluded, but now we just don't support it
            magazine::resetInternalData(m_res, desc);
        }
    }

    // assuming only one source node
    auto &src_op_info = m_script[0];
    // random command
    cv::gimpl::stream::Cmd cmd;
    // push something to the input data queues of the source node
    for (auto &&in_eh : src_op_info.nh->inEdges())
    {
        ade::TypedGraph<DataQueue> qgr(m_g);
        if (qgr.metadata(in_eh).contains<DataQueue>())
        {
            printf("pushing something to the input data queues of the source "
                   "node\n");
            qgr.metadata(in_eh).get<DataQueue>().q.push(cmd);
        }
    }

    nodeActorThread(m_script.back());
    printf("last node called\n");
    for (auto &t : m_threads)
        t.join();
    m_threads.clear();
    // OpenCV backend execution is not a rocket science at all.
    // Simply invoke our kernels in the proper order.
    /*
    GConstGRTModel gcm(m_g);
    for (auto &op_info : m_script)
    {
        const auto &op = m_gm.metadata(op_info.nh).get<Op>();

        // get input data queues for this node handle.
        ade::TypedGraph<DataQueue> qgr(m_g);
        for (auto &&in_eh : op_info.nh->inEdges())
        {
            if (qgr.metadata(in_eh).contains<DataQueue>())
            {
                printf("poping input command\n");
                qgr.metadata(in_eh).get<DataQueue>().q.pop(cmd);
            }
        }
        printf("popped all input command\n");

        // Obtain our real execution unit
        // TODO: Should kernels be copyable?
        GRTKernel k = gcm.metadata(op_info.nh).get<Unit>().k;

        // Initialize kernel's execution context:
        // - Input parameters
        GRTContext context;
        context.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args, std::back_inserter(context.m_args),
                             std::bind(&GRTExecutable::packArg, this, _1));

        // - Output parameters.
        // FIXME: pre-allocate internal Mats, etc, according to the known meta
        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            // FIXME: Can the same GArg type resolution mechanism be reused
            // here?
            const auto out_port = ade::util::index(out_it);
            const auto out_desc = ade::util::value(out_it);
            context.m_results[out_port] = magazine::getObjPtr(m_res, out_desc);
        }

        // Now trigger the executable unit
        k.apply(context);

        // As Kernels are forbidden to allocate memory for (Mat) outputs,
        // this code seems redundant, at least for Mats
        // FIXME: unify with cv::detail::ensure_out_mats_not_reallocated
        // FIXME: when it's done, remove can_describe(const GMetaArg&, const
        // GRunArgP&) and descr_of(const cv::GRunArgP &argp)
        for (const auto &out_it :
             ade::util::indexed(op_info.expected_out_metas))
        {
            const auto out_index = ade::util::index(out_it);
            const auto expected_meta = ade::util::value(out_it);

            if (!can_describe(expected_meta, context.m_results[out_index]))
            {
                const auto out_meta = descr_of(context.m_results[out_index]);
                util::throw_error(
                    std::logic_error("Output meta doesn't "
                                     "coincide with the generated meta\n"
                                     "Expected: " +
                                     ade::util::to_string(expected_meta) +
                                     "\n"
                                     "Actual  : " +
                                     ade::util::to_string(out_meta)));
            }
        }

        // (?) each out edge has inEdges and outEdges?
        // And outEdges are the inEdges of next node?
        for (auto &&out_eh : op_info.nh->outNodes())
        {
            for (auto &&out_eeh : out_eh->outEdges())
            {
                if (qgr.metadata(out_eeh).contains<DataQueue>())
                {
                    qgr.metadata(out_eeh).get<DataQueue>().q.push(cmd);
                    printf("pushed output command\n");
                }
            }
        }
    } // for(m_script)
*/

    for (auto &it : output_objs)
        magazine::writeBack(m_res, it.first, it.second);
}
